import os
import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertForSequenceClassification, BertTokenizer

PAD_TOKEN_LABEL_ID = CrossEntropyLoss().ignore_index
LEARNING_RATE_MODEL = 1e-5
LEARNING_RATE_CLASSIFIER = 1e-3
WARMUP_STEPS = 0
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
SEED = 42
NO_CUDA = False


class SentimentBERT:
    model = None
    tokenizer = None

    def __init__(self):
        self.pad_token_label_id = PAD_TOKEN_LABEL_ID
        self.device = torch.device("cuda" if torch.cuda.is_available() and not NO_CUDA else "cpu")

    def predict(self, dataloader):
        if self.model is None or self.tokenizer is None:
            self.load()

        preds = self._predict_tags_batched(dataloader)
        return preds

    def evaluate(self, dataloader):
        y_pred = self._predict_tags_batched(dataloader)
        y_true = np.append(np.zeros(12500), np.ones(12500))

        return classification_report(y_true, y_pred)

    def _predict_tags_batched(self, dataloader):
        preds = []
        self.model.eval()
        for batch in tqdm(dataloader, desc="Computing NER tags"):
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                outputs = self.model(batch[0])
                _, is_neg = torch.max(outputs[0], 1)
                preds.extend(is_neg.cpu().detach().numpy())

        return preds
    
    def train(self, model, device, train_loader, optimizer, total_train_loss, epoch):

        #Model
        train_epoch_iterator = tqdm(train_loader, desc="Iteration")
        acc = 0.
        train_loss = 0.
        total = 0
        t0 = time.time()

        for step, batch in enumerate(train_epoch_iterator):
            model.train()
            batch = tuple(t.to(self.device) for t in batch)
            outputs = model(batch[0], labels=batch[1])
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            #Performacne metrics 
            total += target.size(0)
            train_loss += loss.sum().item()
            loss.backward()

            total_train_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
        
        print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f}".format(epoch, time.time()-t0,train_loss/total))
        return train_loss/total, total_train_loss

    
    def test(self, model, device, test_loader, optimizer, total_test_loss, epoch):

        #Model
        test_epoch_iterator = tqdm(test_loader, desc="Iteration")
        acc = 0.
        test_loss = 0.
        total = 0
        t0 = time.time()

        for step, batch in enumerate(test_epoch_iterator):
            model.train()
            batch = tuple(t.to(self.device) for t in batch)
            outputs = model(batch[0], labels=batch[1])
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS

            #Performacne metrics 
            total += target.size(0)
            test_loss += loss.sum().item()
            loss.backward()

            total_test_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1
        
        print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f}".format(epoch,time.time()-t0,train_loss/total))
        return train_loss/total, total_test_loss

    
    def model_fit_evaluate(self, tokenizer, train_loader, test_loader, model, epochs):
        
        '''Set up model. Data trained in batches and repeated for number of epochs '''

        #Model
        assert self.model is None  # make sure we are not training after load() command
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

        #Model params
        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": LEARNING_RATE_MODEL},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

        #Training params
        acc = 0.
        train_loss = 0.
        total = 0
        t0 = time.time()
        t_total = len(train_loader) // GRADIENT_ACCUMULATION_STEPS * epochs
        global_step = 0
        logging_loss = 0.0
        total_train_loss, total_test_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(epochs, desc="Epoch")
        self._set_seed()

        #Performance metrics
        best_acc = 0 
        model_loss={}
        model_loss['train_loss']=[]

        # Train
        print("***** Running training *****")
        print("Training on %d examples", len(train_loader))
        print("Num Epochs = %d", epochs)
        print("Total optimization steps = %d", t_total)

        #Iterate through epochs 
        for _ in train_iterator:
            train_epoch_iterator = tqdm(train_loader, desc="Iteration")
            train_loss, total_train_loss = train(model, device, train_loader, optimizer, total_train_loss, epoch)
            #Test loss
            test_loss, total_test_loss = test(model, device, test_loader, optimizer, total_test_loss, epoch)
            model_loss['train_loss'].append(train_loss)
            model_loss['test_loss'].append(test_loss)
        
        #Plot
        plot_loss(model_loss)
        self.model = model

        return global_step, total_train_loss / global_step

    def train_full(self, tokenizer, dataloader, model, epochs):
        
        assert self.model is None  # make sure we are not training after load() command
        model.to(self.device)
        self.model = model
        self.tokenizer = tokenizer

        #Training params
        acc = 0.
        train_loss = 0.
        total = 0
        t0 = time.time()

        t_total = len(dataloader) // GRADIENT_ACCUMULATION_STEPS * epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        optimizer_grouped_parameters = [
            {"params": model.bert.parameters(), "lr": LEARNING_RATE_MODEL},
            {"params": model.classifier.parameters(), "lr": LEARNING_RATE_CLASSIFIER}
        ]
        optimizer = AdamW(optimizer_grouped_parameters)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=t_total)

        # Train
        print("***** Running training *****")
        print("Training on %d examples", len(dataloader))
        print("Num Epochs = %d", epochs)
        print("Total optimization steps = %d", t_total)

        #Training metrics 
        global_step = 0
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()
        train_iterator = trange(epochs, desc="Epoch")
        self._set_seed()

        for _ in train_iterator:
            epoch_iterator = tqdm(dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                outputs = model(batch[0], labels=batch[1])
                loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                total += target.size(0)
                train_loss += loss.sum().item()
                #Accuracy - torch.eq computes element-wise equality
                acc += pred.eq(target.view_as(pred)).sum().item()

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

                    scheduler.step()  # Update learning rate schedule
                    optimizer.step()
                    model.zero_grad()
                    global_step += 1

        self.model = model

        print("\nEpoch {}: \nTime Usage:{:4f} | Training Loss {:4f} | Acc {:4f}".format(epoch,time.time()-t0,train_loss/total,acc/total))
  
        return global_step, tr_loss / global_step

    def plot_loss(model_history):
        'Plot training loss'
        plt.figure(figsize=(10,4))
        plt.plot(model_history['train_loss'], color='r')
        plt.plot(model_history['test_loss'], color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Loss Function')
        plt.legend(['Training', 'Validation'])
        plt.show()
    

    def _set_seed(self):
        torch.manual_seed(SEED)
        if self.device == 'gpu':
            torch.cuda.manual_seed_all(SEED)

    def load(self, model_dir='weights/'):
        if not os.path.exists(model_dir):
            raise FileNotFoundError("folder `{}` does not exist. Please make sure weights are there.".format(model_dir))

        self.tokenizer = BertTokenizer.from_pretrained(model_dir)
        self.model = BertForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
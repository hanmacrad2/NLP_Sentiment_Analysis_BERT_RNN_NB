#BERT
print('Start')
import os
import argparse

from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from data_prep import SentimentDataset
from model import SentimentBERT

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews

parser = argparse.ArgumentParser(prog='script')
parser.add_argument('--train', action="store_true", help="Train new weights")
parser.add_argument('--evaluate', action="store_true", help="Evaluate existing weights")
parser.add_argument('--predict', default="", type=str, help="Predict sentiment on a given sentence")
parser.add_argument('--path', default='weights/', type=str, help="Weights path")
parser.add_argument('--train-file', default='../data/train.csv',
                    type=str, help="Amazon reviews train file.")
parser.add_argument('--test-file', default='../data/text.csv',
                    type=str, help="Amazon reviews test file.")
args = parser.parse_args()

def model_fit_evaluate(train_file, test_file, epochs=20, output_dir="weights/"):

    #Configuration + Model
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    #Data
    dt = SentimentDataset(tokenizer)
    train_loader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
    test_loader = dt.prepare_dataloader(test_file, sampler=RandomSampler)
    predictor = SentimentBERT()
    predictor.model_fit_evaluate(tokenizer, train_loader, test_loader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def train(train_file, epochs=20, output_dir="weights/"):

    #Configuration + Model
    config = BertConfig.from_pretrained(BERT_MODEL, num_labels=NUM_LABELS)
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL, do_lower_case=True)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, config=config)

    #Data
    dt = SentimentDataset(tokenizer)
    dataloader = dt.prepare_dataloader(train_file, sampler=RandomSampler)
    predictor = SentimentBERT()
    predictor.train(tokenizer, dataloader, model, epochs)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate(test_file, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader(test_file)
    score = predictor.evaluate(dataloader)
    print(score)


def predict(text, model_dir="weights/"):
    predictor = SentimentBERT()
    predictor.load(model_dir=model_dir)

    dt = SentimentDataset(predictor.tokenizer)
    dataloader = dt.prepare_dataloader_from_examples([(text, -1)], sampler=None)   # text and a dummy label
    result = predictor.predict(dataloader)

    return "Positive" if result[0] == 0 else "Negative"


""" if __name__ == '__main__':
    if args.train:
        os.makedirs(args.path, exist_ok=True)
        train(args.train_file, epochs=10, output_dir=args.path)

    if args.evaluate:
        evaluate(args.test_file, model_dir=args.path)

    if len(args.predict) > 0:
        print(predict(args.predict, model_dir=args.path)) """

    #print(predict("It was truly amazing experience.", model_dir=args.path))

if __name__ == '__main__':
    
    print('Train')
    os.makedirs(args.path, exist_ok=True)
    model_fit_evaluate(args.train_file, args.test_file, epochs=10, output_dir=args.path)

    print('Test')
    evaluate(args.test_file, model_dir=args.path)

    print('Predict')
    print(predict(args.predict, model_dir=args.path))
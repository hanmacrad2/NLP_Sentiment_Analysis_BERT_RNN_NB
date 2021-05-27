#Imports
print('Start')
from train_bert_model import Train_BERT

def main():

    print('Load train file')
    obj = Train_BERT('train_formatted.csv', batch_size=8, labels=[1,0], max_seq_len=40, epochs=15)

    # object initilization fails if the filename is incorrect
    # you can also set the seq length,batch size and labels you want to train explicitly

    obj.max_seq_len = 50
    obj.batch_size = 32
    obj.labels = [1,0]
    obj.initilize_model()
    obj.start_train()
    obj.evaluate_model()

if __name__ == "__main__":
    main()
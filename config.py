device = 'cpu'
model_name ='csebuetnlp/banglabert'
max_seq_length = 30
categories = ['Economy', 'Education', 'Entertainment', 'International',
       'Politics', 'National', 'Science_Technology', 'Sports']
num_classes = len(categories)
dropout = 0.5
mean_pool = True
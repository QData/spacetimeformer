About Input Output File format 
=======================================

+ We provide utility code to read in FASTA string input format
  - See example data files @ [Here](https://github.com/QData/spacetimeformer/tree/master/data)
  - Also below we include two sequence examples in Fasta format 
  - In each example's header line: 1 represents positive class label and 0 represents negative class label (so the task is to classify a string into two classes)
```
>1
MKTPITEAIAAADNQGRFLSNTELQAVNGRYQRAAASLEAARSLTSNAERLINGAAQAVYSKFPYTSQMPGPQYASSAVGKAKCARDIGYYLRMVTYCLVVGGTGPMDEYLIAGLEEINRTFDLSPSWYVEALNYIKANHGLSGQAANEANTYIDYAINALS
>0
PYTINSPSQFVYLSSAYADPVELINLCTNALGNQFQTQQARTTVQQQFADAWKPSPVMTVRFPASDFYVYRYNSTLDPLITALLNSFDTRNRIIEVNNQPAPNTTEIVNATQRVDDATVAIRASINNLANELVRGTGMFNQAGFETASGLVWTTTPAT
```

+ However, the most basic input requriement for spacetimeformer python functional call is just a simple numpy array; You can have whatever string format you prefer... and convert your format to numpy array...  
```
Xtrain = [[1,0,1,0,1], [1,1,1,0,1]]
Xtest = [[1,1,1,1,1], [1,0,1,0,1]]

kernel.compute_kernel(Xtrain, Xtest)

train_kernel = kernel.get_train_kernel()
test_kernel = kernel.get_test_kernel()
```
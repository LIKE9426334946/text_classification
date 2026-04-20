# text_classification
文本二分类

# 运行方法
训练模型  
python train.py  

训练完成后会生成
best_textcnn.pt

预测  
python predict.py

# kaggle运行
%cd /kaggle/working  
!rm -rf /kaggle/working/text_classification  
!git clone https://github.com/LIKE9426334946/text_classification.git  
%cd /kaggle/working/text_classification  
!python3 train.py

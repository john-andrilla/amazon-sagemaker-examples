From dgllib/dgl-sagemaker-gpu:dgl_0.4_mxnet_1.5.1

RUN pip install gluonnlp pandas
RUN pip install spacy 
RUN python3 -m spacy download en
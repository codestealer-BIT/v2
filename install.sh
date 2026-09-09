pip3 install transformers==4.53.0
pip3 install faiss-gpu-cu12
pip3 install byted_tensorproto
pip install yacs
pip install scikit-learn
pip3 install https://luban-source.byted.org/repository/scm/search.nlp.matx_rtc_pip_wheels_1.6.0.12.tar.gz
pip3 install https://luban-source.byted.org/repository/scm/search.nlp.libcut_py_matx4_2.3.0.20.tar.gz
pip3 install https://luban-source.byted.org/repository/scm/nlp.tokenizer.py_1.0.0.115.tar.gz
pip3 install https://luban-source.byted.org/repository/scm/nlp.lib.ptx2_1.0.0.828.tar.gz

bvc clone search/nlp/libcut_model_ml_20210122 /opt/tiger/libcut_model_ml_20210122 --version 1.0.0.2
bvc clone search/nlp/mlcut_files /opt/tiger/libcut_model_ml_20201229 --version 1.0.0.4
pip3 uninstall -y byted-matxscript
pip3 install byted-matxscript==1.8.2 --index=https://bytedpypi.byted.org/simple/  --trusted-host=bytedpypi.byted.org
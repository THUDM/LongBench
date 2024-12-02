pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 
pip install Cython
pip install packaging
echo "Installing the required Python packages..."
pip install nemo_toolkit[all] --user
pip install flask
pip install flask_restful
pip install sshtunnel_requests
pip install tritonclient[all]
pip install wonderwords
pip install openai
pip install tiktoken
pip install tenacity
pip install transformers==4.42.4
# - pip install transformers==4.44.2
# - pip install transformers==4.46.2
pip install flash-attn==2.5.6
pip install accelerate
pip install vllm==0.4.0.post1
pip install huggingface_hub
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
pip install html2text
pip install bs4
pip install pandas
pip install google-generativeai
echo "reinstalling torch"
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 
echo "Installing the FlashAttention package..."
pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary --no-deps
pip install numpy==1.23.5 --no-deps
pip install huggingface_hub==0.23.2 --no-deps
pip install nltk==3.8.1 
pip install regex 
pip install pyyaml
pip install tqdm
pip install hydra-core
pip install omegaconf
pip install pytorch-lightning 
pip install numpy==1.23.5 --no-deps
pip install huggingface_hub==0.23.2 --no-deps
pip install fuzzywuzzy
pip install rouge
echo "Installation complete."
export only_last_logits=1
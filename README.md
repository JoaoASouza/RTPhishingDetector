# Real-Time Phising detector

Ferramenta de treinamento e avaliação de cGANs para geração de dados sintéticos para o contexto de detecção de malwares Android.

## Preparação e Execução

### 1. Clonar o repositório 
   ```bash
    git clone https://github.com/JoaoASouza/RTPhishingDetector.git
    cd RTPhishingDetector
    ```

### 2. Instalar dependências
    ```bash
    pip3 install -r requirements.txt
    ```

    ou

    ```bash
    pip install -r requirements.txt
    ```

### 3. Executar a ferramenta: 

    ```bash
    python3 classify.py
    ```

    ou

    ```bash
    python classify.py
    ```

## Obtendo novos dados

Para incremento do dataset são necessários 3 etapas:

### 1. Obter URLs de phishing a partir das blacklists
    ```bash
    python3 get_phishing_URLs.py
    ```
    As URLs obtitas ficarão salvas no arquivo "phishing_urls_phishtank_{data_atual}" dentro do diretório "lists"

### 2. Obter URLs legítimas
    Primeiramente é necessário baixar a página html do MOZ disponível em https://moz.com/top500 e substituir o arquivo "moztop500.html"
    Em seguida deve-se executar o script
    ```bash
    python3 get_legitimate_URLs.py
    ```
    As URLs obtitas ficarão salvas no arquivo "legitimate_urls_moz_{data_atual}" dentro do diretório "lists"

### 3. Obter os atributos das URLs
    ```bash
    python3 get_URL_attrs.py
    ```
    Dentro do arquivo "get_URL_attrs.py" modificar a flag LEGITIMATE_DATA para True caso se deseje obter os dados das URLs legítimas ou para False no caso de URLs de phishing
    Caso se deseje utilizar o conjunto completo de fetures modificar a flag USE_EXTERN_TOOLS para True
    Os dados ficarão salvos nos arquivos "out{data_atual}.csv" ou "out{data_atual}_legitimate.csv" (dependendo da flag LEGITIMATE_DATA) dentro do diretório "datasets"

### 4. Executar o script para unir os todos os arquivos do diretório "datasets"
    ```bash
    python3 build_dataset.py
    ```

## Ambientes de teste

A ferramenta foi executada e testada na prática nos seguintes ambientes:

1. Linux Mint 20.1 x86_64<br/>
   Kernel Version = 5.4.0-155-generic<br/>
   Python = 3.9.13 <br/>
   Módulos Python conforme [requirements](requirements.txt).
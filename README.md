# tiktok-popularity-analysis
Projeto da disciplina SME0852 - Prática em Ciência de Dados para análise e extração de valor acerca da popularidade de músicas do Tiktok.

## Configurando um ambiente virtual e um kernel do Jupyter

De modo a garantir a execução do código sem conflitos com as bibliotecas localmente instaladas,
uma prática recomendada é a criação de um ambiente virtual:

```bash
python -m venv .venv
```

Uma vez que o ambiente esteja criado, ele deve ser ativado:

```bash
source .venv/bin/activate
```

Para instalar as dependências do projeto, basta executar, depois de criar o ambiente
e ativá-lo:

```bash
pip install -r requirements.txt
```

O próximo passo é criar um kernel. Para tanto, execute, dentro do ambiente criado, o comando:

```bash
python -m ipykernel install --user --name=tiktok-analysis
```

e, dentro do notebook, selecione o kernel conforme nomeado acima: *tiktok-analysis*.

Caso precise sair do ambiente virtual, basta executar, de dentro do ambiente, o comando:

```bash
deactivate
```
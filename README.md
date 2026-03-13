### Описание

Демонстрация работы модели **GP-MoLFormer**.

Оригинальная реализация модели:
[https://github.com/IBM/gp-molformer](https://github.com/IBM/gp-molformer)

В репозитории добавлен ноутбук с примером запуска модели и демонстрацией генерации молекул на примере аспирина.

---

# Установка

Клонировать репозиторий:

```bash
git clone https://github.com/gvo104/molformer-demo.git
cd molformer-demo/gp-molformer
```

Создать окружение:

```bash
conda env create -f environment.yml
conda activate gp-molformer
```

Если RDKit не установился:

```bash
conda install -c conda-forge rdkit
```

---

# Запуск

Вернуться в корень проекта:

```bash
cd ..
jupyter notebook
```

Открыть:

```
test.ipynb
```

---

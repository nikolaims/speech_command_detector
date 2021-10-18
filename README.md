# Решение Тестового задания в отдел “Data Analytics”
Выполнил **Николай Сметанин**


## 1. Приложение для детектирования слова STOP

В качестве шаблона для детектирования было выбрано слово **STOP** в английском произношении **[stäp]**. Приложение 
реализовано на языке python. Ниже приведены инструкции по установке и использованию приложения

### 1.1. Установка
1. Установить менеджер пакетов conda, например 
   [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
1. Склонировать репозиторий и перейти в корневую папку
1. Создать среду в соответствии с 
   [`environment.yml`](https://github.com/nikolaims/siemens_test_task/blob/master/environment.yml):
    ```
    conda env create -f environment.yml
    ```
1. Активировать среду 
   ```
   conda activate siemens_test_task
   ```
1. Установить пакет 
   [`solution`](https://github.com/nikolaims/siemens_test_task/blob/master/solution):
   ```
   pip install -e .
   ```

### 1.2. Использование
Приложение поддерживает 3 разных режима работы:
1.  Декодирование из `.wav` файла и визуализация разметки аудио:
    ```
    python app.py file PATH
    ```
    В аргументе `PATH` необходимо указать путь к аудио файлу. Поддерживаются моно и стерео записи в формате `.wav`
    с произвольной частотой дискретизации. Для анализа берется только первый канал, а частота дискретизации 
    ресемплируется к 16kHz. Примеры записей находятся в 
     [`audio_samples`](https://github.com/nikolaims/siemens_test_task/blob/master/audio_samples)
    
2. Запись аудио с микрофона, с последующим декодированием и визуализацией:
   ```
   python app.py mic -r SEC
   ```
   В аргументе `SEC` необходимо указать длину записи в секундах. Запись начнется после вывода надписи *ON AIR*.
3. Декодирование и визуализация аудиопотока из микрофона в режиме реального времени:
   ```
   python app.py mic
   ```
   
### 1.3. Пример
В качестве примера приведена команда для декодирования из файла 
[`audio_samples/one_stop_three.wav`](https://github.com/nikolaims/siemens_test_task/blob/master/audio_samples/one_stop_three.wav)
в котором последовательно произносятся слова ONE, STOP, TREE:
    ```
    python app.py file audio_samples/one_stop_three.wav
    ```
Результат приведен на рис 1.:
![alt text](images/one_stop_tree.png)
*Fig. 1. STOP spotting on one_stop_tree.wav*
   
## 2. Методы
### 2.1. Данные
Задача, которую предлагалась решать часто обозначается как keyword spotting problem. В качестве данных для обучения 
моделей был выбран один из самых популярных датасетов для этой задачи - 
[Speech Commands Dataset](https://paperswithcode.com/dataset/speech-commands), содержащий записи набора слов, 
произнесенные разными людьми, а также записи с фоновым шумом. В качестве целевого слова было выбрано 
слово **STOP [stäp]**. Все записи из Speech Commands Dataset слова STOP были использованы и составили 20% от конечного 
датасета. Остальные слова были выбраны так, чтобы составлять 45% записей конечного датасета. Записи фонового шума без 
слов составили 35% конечного датасета. Сохраняя пропорции [слова STOP, другие слова, фон] датасет был разбит на выборки 
train, validation и test в соотношении 60%, 20% и 20% соответственно. 

### 2.2. Модель

В качестве декодирующего алгоритма была выбрана **сверточная нейронная сеть**, на вход которой подается спектрограмма 
отрезка аудиозаписи длины **1 секунда** с частотой дискретизации **16kHz**. Свертка по частотно-временному представлению
позволяет обеспечить:
 1. Обобщение относительно момента времени когда было произнесено целевое слово
 2. Обобщение относительно высоты голоса говорящего  

Реализация нейронной сети и её обучение реализовано при помощи pytorch. В качестве архитектуры сети была выбрана сеть 
с двумя сверточными слоями и одним полносвязным слоем на выходе. Архитектура модели подобна архитектуре benchmark-модели
использованной в статье с описанием [Speech Commands Dataset](https://paperswithcode.com/dataset/speech-commands). 
Главным отличием модели настоящей работы от приведенной в статье является адаптация к решению бинарной классификации 
(сегмент аудио содержит либо не содержит слово STOP) вместо классификации нескольких классов слов. Модель задана в виде 
класса, подробные параметры которой можно посмотреть в
 [`solution.model`](https://github.com/nikolaims/siemens_test_task/blob/15b5861578199a69e77839cb443f8ef20249d93a/solution/model.py#L4-L18).

### 2.3. Обучение
В качестве loss-function использовалась Binary Cross Entropy. Скрипт реализующий обучение - 

[`train.py`](https://github.com/nikolaims/siemens_test_task/blob/master/train.py). Обучение происходило эпохами по 
160 батчей состоящих из 64 семпла. 

### 2.4. Качество
Поскольку в задании не определен контекст задачи метрикой качества модели был выбран 
[Matthews correlation coefficient](https://en.wikipedia.org/wiki/Matthews_correlation_coefficient)
 как метрику общего вида применяемой для бинарной классификации на несбалансированной выборке. На картинке 2A ниже 
приведено качество модели после разного количества эпох на выборке valid и финальная метрика на test. На картинке 2B
представлена кривая precision-recall для финальной модели на тестовой выборке, в зависимосте от потребностей задачи 
может быть выбран тот или иной порог ответа сети. 

![alt text](images/learning_and_pr_curves.png)
*Fig. 2. Learning (A) and precision-recall (B) curves*

### 2.5 Скользящее окно
Для применения модели к аудио длительности больше 1 секуды была применена техника скользящего окна. 
Ответ модели был взвешен на каждом окне при помощи tukey window. Один временной отсчет в результате содержит несколько 
взвешенных ответов, которые суммируются и делятся на сумму весов. 

### 2.5. Ограничения модели
Основные ограничения модели
- определенный темп речи, два близко расположенных слова могут попасть в 
одно окно что затруднит декодирование; 
- один говорящий - модель не разделяет говорящих одновременно с одинаковой громкостью людей;
- ошибки в очень тихих записях - необходимо добавить voice activity detector.

## 3. Структура репозитория
Методы, классы и константы реализованы в виде пакета `solution` и разбиты по модулям в зависимости от функциональности:
* [`solution.data`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/data.py) - работа с данными
* [`solution.infer`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/infer.py) - использование 
  обученной модели как по отрезкам 1 секунда так и скользящим окном для длинной записи
* [`solution.learning`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/learning.py) - helpersы для обучения
* [`solution.mic_rt`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/mic_rt.py) - онлайн мониторинг сигнала микрофона, детектирование и отрисовка
* [`solution.model`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/model.py) - определение модели
* [`solution.preprocessing`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/preprocessing.py) - 
  задает обработку семплов до входа модели включая переход в частотно-временное представление
* [`solution.utils`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/utils.py) - вспомогательные функции для работы с аудио
* [`solution.viz`](https://github.com/nikolaims/siemens_test_task/blob/master/solution/viz.py) - вспомогательные функции для визуализации

  Репозиторий содержит следующие скрипты:
* [`app.py`](https://github.com/nikolaims/siemens_test_task/blob/master/app.py) - CLI приложения
* [`evaluate.py`](https://github.com/nikolaims/siemens_test_task/blob/master/evaluate.py) - измерение качества моделей
* [`train.py`](https://github.com/nikolaims/siemens_test_task/blob/master/train.py) - обучение модели

Также репозиторий  включает следующие директории:
* [`audio_samples`](https://github.com/nikolaims/siemens_test_task/blob/master/audio_samples) - примеры аудио
* [`images`](https://github.com/nikolaims/siemens_test_task/blob/master/images) - визуализация результатов
* [`model_states`](https://github.com/nikolaims/siemens_test_task/blob/master/model_states) - сохраненные параметры модели
* [`ref_datasets`](https://github.com/nikolaims/siemens_test_task/blob/master/ref_datasets) - содержит csv таблицы с 
  ссылками на семплы, а также их разделение на выборки
  
## 4. TODO 
Время потраченное на работу и оформление ~ 30 часов. Далее работа может быть продолжена следующим образом:
1. Оформить документацию пакета solution
2. Оценка качества работы алгоритма со скользящим окном для потока аудио
3. Добавление voice activity detector - нет необходимости производить декодирование если отсутствует речь
4. Улучшение модели и сравнение с текущей




  


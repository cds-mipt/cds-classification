# cds-classification

## crop_light.py
Этот скрипт используется для получения датасета из светофоров.   
На вход он принимает следующие параметры:
* `folder-train`
* `folder-test`
* `attitude`
* `input-file`

`folder-train` - путь папки в которую будет записываться датасет для обучения сетки, если папка не существует, то она будет создана автоматически   
`folder-test` - путь папки в которую будет записываться датасет для валидации сетки, если папка не существует, то она будет создана автоматически   
`attitude` - отношение `train` к `train+test` (например: при `attitude=0.7` 70% информации будет записано в `folder-train`) дефолтное значение `attitude` равняется `0.8` 
`input-file` - файл типа `JSON` в котором находятся аннотации к городу/городам/всему_датасету   

### Пример использования
`python crop_light.py --output-folder-test /datasets/DTLD/DTLD_crop/test --output-folder-train /datasets/DTLD/DTLD_crop/train --attitude 0.9 --input-file /datasets/DTLD/JSONS/Bochum_all.json`

#### Важно! Для корректной работы в JSON файлах должны быть прописаны правильные `path` файлов в аннотациях




## dataset_maker.py
скрипт предназначен для создания одного общего датасета из множества других. В качестве аргумента принимает путь конфигурационного файла: 

`python dataset_maker.py --config-path /path/to/your/config`

пример конфигурационного файла есть в репозитории `cds-classification`


конфигурационный файл состоит из трёх разделов: `destinations`, `subfolders`, `params`

в `destination` есть два подраздела: `source_folders` для перечисления папок, из которых будет собираться датасет, и `target_folder` папка, в которую будет собираться датасет


в `subfolders` есть `map_folders` в котором нужно определить словарь, в котором в ключи будут являться названиями папок в `target_folder`, а в качестве значений будет list из папок `source_folders`


#### пример словаря:
`map_folders = {'G' :['G'], 'NA':['NA'], 'NT':['NT'], 'R':['R','YR'], 'TR':['TR'], 'Y':['Y']}`


в `params` есть два раздела: `train_test_split` для указания отношения `train/(train+test)` и `rotate` может принимать значения 0 или 1, в зависимости хотим мы, чтоб все изображения были вертикальными или нет


### ВАЖНО
для правильного поворота файлы должны иметь валидные имена

## test.py
Этот скрипт предназначен тестирования сети ResNetM

пример:


`python test.py --device 2 --test-dir /path/to/test/dir/ --model /path/to/model/ --error-logs 1`


`device` отвечает за номер gpu, на котором будет тестироваться сеть


в `test-dir` следует указать путь до папки, в которой тестовая выборка будет разбита на подпапки

`--error-logs` отвечает за логи ошибок, если установлена 1, то логи будут сохранятся. Создаётся папка `error_logs` в ней папка с датой теста, а в ней две папки `false_negative` и `false_positive`, в первой лежат изображения, ошибочно отнесенные к другим классам, во второй лежат изображения, ошибочно отнесенные к этому классу.


test.py печатает получившийся precision, recall, fscore, accuracy и fps

пример вывода:

![test output](https://github.com/cds-mipt/cds-classification/blob/master/Screenshot%20from%202020-07-01%2022-05-29.png)


predict*


## train.py
Этот скрипт предназначен обучения сети ResNetM

пример:


`python train.py --device 2 --train-dir /path/to/train/dir/ --val-dir /path/to/val/dir/ --callback-folder ./models/`


`device` отвечает за номер gpu, на котором будет обучаться сеть


в `train-dir` следует указать путь до папки, в которой обучающая выборка будет разбита на подпапки


в `val-dir` следует указать путь до папки, в которой валидационная выборка будет разбита на подпапки

папка чекпойнтов задается флагом --callback-folder

папка логов для tensorboard задается флагом --tensorboard-folder

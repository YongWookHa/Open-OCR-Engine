# OCR-Engine

# Overview

Thanks to open source, it's not really difficult to build a customed OCR engine. In this repository, we'll do something below.

1. download and pre-process dataset for training model
2. try training text detection model and text recognization model respectively
3. custom your own OCR-Engine
4. serve your model on web

In [Google OCR Service Paper](https://das2018.cvl.tuwien.ac.at/media/filer_public/85/fd/85fd4698-040f-45f4-8fcc-56d66533b82d/das2018_short_papers.pdf#page=23), we can get a hint of building OCR Engine.

![](https://www.dropbox.com/s/zjkvt6cm3pv2f7x/google_ocr_structure.jpg?raw=1)

_In this repository, we will leave `Direction ID`, `Script ID` and `Layout Analysis` parts empty._

# Getting Started

## 1. Data Generation

First of all, we need to prepare training data. If you don't have good quality data, you can generate one. There are three steps to go.

### A) Collect corpus.

Locate your corpus in `./generate_data/texts/` directory. This corpus will be tokenized and renderd in the images of dataset. So, it would be best to gather corpus in target domain.
I recommend you to prepare more than 1MB of corpus as `.txt` file.

_[Get Corpus](https://lionbridge.ai/datasets/the-best-25-datasets-for-natural-language-processing/)_

### B) Collect fonts.

Locate your font files in `./generate_data/fonts/<lang>/` directory. The extension of font files should be `.otf` or `.ttf`. **Separate fonts by languages.** If your language is English the `<lang>` folder can be `en`.

_[Get Fonts](https://www.dafont.com/)_

### C) Generate line data.

We will generate line image like below and `.pkl` files which contains location of every character in the image. A `pkl` file is created for each image. Additionally, total ground truth data will be generated in `gt.pkl` file.

![](https://www.dropbox.com/s/a95xi3xszdq5qlo/generated_line_0.jpg?raw=1)

This line data is ingredients for making _paragraph_ dataset. _(see step **D)**)_

```
> cd generate_data
> python run.py -i texts/my-corpus.txt \
    -l ko -nd -c 10000 -f 200 \
    -rs -w 20 -t 1 -bl 2 -rbl -k 1 -rk -na 2 \
    --output_dir out/line
```

- `-i` : input corpus
- `-l` : language of fonts (language name in `generate_data/fonts` directory)
- `-c` : number of lines to be used for generating data
- _You can check all options in `generate_data.py`_

+) If you put `--bbox` option, you can visualize the bounding box of all characters. The image samples below are include bounding box visualization. You shouldn't put this option for training data.

### D) Merge line data to paragraph.

To train text detection model, we will merge line data which we already generated above to paragraph. You can use `merge_lines.py` code in `generate_dataset` directory.

```
> cd generate_data
> python merge_lines.py -o vertical -b out/line --width 2000 --height 1000 --min 1 --max 5
```

then, you will get paragraph data and `out/line/combined/merged_gt.pkl` data below.

![](https://www.dropbox.com/s/m06dnj5m85y5zwy/generated_1.jpg?raw=1)

![](https://www.dropbox.com/s/5v90hlyuafqibj4/generated_0.jpg?raw=1)

### E) Crop word data.

To train text recognition model, we will generate word data by cropping paragraph data which we made in **C)**. 

```
> cd generate_data
> python crop_words.py --pickle out/lines/combined/merged_gt.pkl \
    --image_dir out/lines/combined --output_dir out/words
```

Then you can get word-level-splited cropped data. The total data would be located in `out/words/gt.pkl` with command above.

![](https://www.dropbox.com/s/b91q68iw9j78ctj/generated_data_word_0.png?raw=1) ![](https://www.dropbox.com/s/ay4zt6keklq696f/generated_data_word_1.png?raw=1) ![](https://www.dropbox.com/s/kjrwyn0n0feyeym/generated_data_word_2.png?raw=1)   

_Okay. We finished preparing dataset for training._

## 2. Train Text Detection Model

There are several hyper-parameters of text detection model in `settings/default.yaml`. I don't recommend you to edit them without knowledge of specific element.

```
> python train.py -m detector \
    --data_path generate_data/out/line/combined/merged_gt.pkl \
    --version 0 --batch_size 4 --learning_rate 5e-5 \
    --max_epoch 100 --num_workers 4
```

To monitor the training progress, use tensorboard.

```
> tensorboard --logdir tb_logs
```

![](https://www.dropbox.com/s/dxky1qf1oz83v20/craft_train_log.jpg?raw=1)

## 3. Train Text Recognition Model

Text Recognizer also has some hyper-parameters. Thanks to [deep-text-recognition-benchmark](https://github.com/clovaai/deep-text-recognition-benchmark), It's really easy to change parts which consist recognzier. 

### Modules

`Transformation`: select Transformation module [None | TPS].  
`FeatureExtraction`: select FeatureExtraction module [VGG | RCNN | ResNet].  
`SequenceModeling`: select SequenceModeling module [None | BiLSTM].  
`Prediction`: select Prediction module [CTC | Attn].  

```
> python train.py -m recognizer --data_path generate_data/out/words/gt.pkl \
    --version 1 --batch_size 64 --learning_rate 1.0 --max_epoch 100 --num_workers 4
```

You need to train the model more than 15k `total iteration`.  

_`iteration per one epoch` = `train_data_size` / ( `batch_size` * `num_gpu` )_  
_`total iteration` = `iteration per one epoch` * `total epoch`_  

You can monitor the training progress with tensorboard as well.  

```
> tensorboard --logdir tb_logs
```

![](https://www.dropbox.com/s/4ye357otthla0c9/DTR_train_log.jpg?raw=1)  
_In the log screenshot, accuracy calculated by exact match cases._  

## 4. Serve OCR Engine with API

Okay, It's time to deploy your OCR-Engine. Before run API server, let's modify some hyper-parameters for prediction stage. Decreasing each thresholds would be better for most test cases.

```yaml
# settings/default.yaml

craft:
    THRESHOLD_WORD : 0.4
    THRESHOLD_CHARACTER: 0.4
    THRESHOLD_AFFINITY: 0.2
```

Then, start API server with `demo.py`. Specify each checkpoints you trained with parameters.

```
> python demo.py --host 127.0.0.1 --port 5000 \ 
    --detector_ckpt <detector checkpoint path> \
    --recognizer_ckpt <recognizer checkpoint path> \
    --vocab vocab.txt
```

Then your OCR-Engine server has been started. 

You can send API request by using `request.py`.

```
> python request.py <img path>
```

Then you will get text and coordinate by response.

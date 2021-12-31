# 1. Data Generation

First of all, we need to prepare training data. If you don't have good quality data, you can generate one. There are three steps to go.

## A) Collect corpus.

Locate your corpus in `./generate_data/texts/` directory. This corpus will be tokenized and renderd in the images of dataset. So, it would be best to gather corpus in target domain.
I recommend you to prepare more than 1MB of corpus as `.txt` file.

_[Get Corpus](https://lionbridge.ai/datasets/the-best-25-datasets-for-natural-language-processing/)_

## B) Collect fonts.

Locate your font files in `./generate_data/fonts/<lang>/` directory. The extension of font files should be `.otf` or `.ttf`. **Separate fonts by languages.** If your language is English the `<lang>` folder can be `en`.

_[Get Fonts](https://www.dafont.com/)_

## C) Generate line data.

We will generate line image like below and `.pkl` files which contains location of every character in the image. A `pkl` file is created for each image. Additionally, total ground truth data will be generated in `gt.pkl` file.

![](https://www.dropbox.com/s/a95xi3xszdq5qlo/generated_line_0.jpg?raw=1)

This line data is ingredients for making _paragraph_ dataset. _(see step **D)**)_

```
> cd generate_data
> python run.py -i texts/my-corpus.txt -l ko -nd -c 10000 -f 200 -rs -w 20 -t 1 -bl 2 -rbl -k 1 -rk -na 2 --output_dir out/line
```

- `-i` : input corpus
- `-l` : language of fonts (language name in `generate_data/fonts` directory)
- `-c` : number of lines to be used for generating data
- _You can check all options in `generate_data.py`_

+) If you put `--bbox` option, you can visualize the bounding box of all characters. The image samples below are include bounding box visualization. You shouldn't put this option for training data.

## D) Merge line data to paragraph.

To train text detection model, we will merge line data which we already generated above to paragraph. You can use `merge_lines.py` code in `generate_dataset` directory.

```
> cd generate_data
> python merge_lines.py -o vertical -b out/line --width 2000 --height 1000 --min 1 --max 5
```

then, you will get paragraph data and `out/line/combined/merged_gt.pkl` data below.

![](https://www.dropbox.com/s/m06dnj5m85y5zwy/generated_1.jpg?raw=1)

![](https://www.dropbox.com/s/5v90hlyuafqibj4/generated_0.jpg?raw=1)

## E) Crop word data.

To train text recognition model, we will generate word data by cropping paragraph data which we made in **C)**. 

```
> cd generate_data
> python crop_words.py --pickle out/lines/combined/merged_gt.pkl --image_dir out/lines/combined --output_dir out/words
```

Then you can get word-level-splited cropped data. The total data would be located in `out/words/gt.pkl` with command above.

![](https://www.dropbox.com/s/b91q68iw9j78ctj/generated_data_word_0.png?raw=1) ![](https://www.dropbox.com/s/ay4zt6keklq696f/generated_data_word_1.png?raw=1) ![](https://www.dropbox.com/s/kjrwyn0n0feyeym/generated_data_word_2.png?raw=1)
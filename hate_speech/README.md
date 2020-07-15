# Hate speech classification
AI Rush 혐오댓글 분류를 위한 경로 입니다. 
Baseline model은 간단한 windowed RNN을 사용하였습니다.

## Repository format
`hate_speech/main.py` 학습 방법과 nsml.bind 함수에 대한 정의
`hate_speech/data.py` Data를 load하는 방법 정의
`hate_speech/model.py` Baseline model 정의
`hate_speech/field.json` Data의 vocab에 대한 정의 (only for torchtext)


## Run experiment

To run the baseline model training, stand in the `airush2020/hate_speech` folder and run 
```
nsml run -e main.py  -m "A good message" -d hatespeech-1
```

## Metric
[F1 Score](https://en.wikipedia.org/wiki/F1_score) 를 사용 합니다.

## Data
개인정보 이슈로 tokenize 이후 numericalize 된 형태로 제공 됩니다.
- tokeninzer
   - 음절 기반 tokenizer
      - 고의적 오탈자와, 신조어가 많은 한국어 댓글 데이터에서는  
      형태소 기반 tokenizer, [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding), [wordpiece tokenizer](https://arxiv.org/pdf/1609.08144.pdf) 가 정상동작 하지 못합니다.
   - vocab 
      - vocab를 이용 역산하여 원문을 밝힐 수 있기에 공개하지 못하였습니다.
      - special tokens  
      UNK: 0, PAD:1, SPACE:2,  BEGIN:3, EOF: 4
      
- e.g.   {"syllable_contents": [3, 32, 218, 12, 25, 2, 205, 337, 16, 2, 113, 9, 2, 558, 195, 16, 2, 113, 17, 68, 2, 288, 51, 39, 12, 25, 4], "eval_reply": 0}   
- 가혹한 제약조건 속에서도 창의적인 도전을 기원합니다.

### Format
See AI Rush dataset documentation.


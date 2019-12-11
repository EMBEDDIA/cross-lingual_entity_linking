# Source code for the cross-lingual version of the "Deep Joint Entity Disambiguation with Local Neural Attention"
Original source code is available at: https://github.com/dalab/deep-ed

## Pre-trained data
Entity embeddings trained with this method using MUSE 300 dimensional pre-trained word vectors (https://github.com/facebookresearch/MUSE). Available [here]().

## Training models
The model was firstly trained on AIDA-train with pre-trained multi-lingual entity embeddings trained on Wikipedia, and then on the wikiann for Croatian, Estonian, Finnish and Slovenian. 
See details of how to run our code below.

### How to run the system and reproduce our results
Follow the same steps 1-14 (https://github.com/dalab/deep-ed) to:
  * process Wikipedia
  * process test datasets
  * train entity embeddings

### Train ED model on English.
mkdir $DATA_PATH/generated/ed_models/
mkdir $DATA_PATH/generated/ed_models/training_plots/
ENTITY_VECS=ent_vecs__ep_375.t7
CUDA_VISIBLE_DEVICES=0 th ed/ed.lua -root_data_dir $DATA_PATH -ent_vecs_filename $ENTITY_VECS -model 'global' -language en 2>&1 | tee logs/log_multi-lingual_train_en

### Test Croatian, Estonian, Finnish and Slovenian datasets with ED model trained only on the English dataset.
ENGLISH_ED_MODEL=model\=global\|language=en\|ep\=201\|lang=en
for language in 'et' 'fi' 'hr' 'sl'
do
    echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
    th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL # > logs/log_test_before_$language
done

### Post-training the English ED model on the Croatian, Estonian, Finnish and Slovenian datasets.
for language in 'et' 'fi' 'hr' 'sl'
do
    rm generated/common_top_words_freq_vectors_fasttext.t7
    CUDA_VISIBLE_DEVICES=0 th ed/ed.lua -root_data_dir $DATA_PATH -ent_vecs_filename $ENTITY_VECS -model 'global' -language $language -load_ed_model $ENGLISH_ED_MODEL 2>&1 | tee logs/log_multi-lingual_train_$language
done

### Test of Croatian, Estonian, Finnish and Slovenian datasets with ED model trained on the English + wikiann datasets.
ENGLISH_ED_MODEL=model\=global\|language=et\|ep\=201\|lang=et
language=et
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_after_$language

ENGLISH_ED_MODEL=model\=global\|language="fi"\|ep\=6\|lang="fi"
language="fi"
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_after_$language

ENGLISH_ED_MODEL=model\=global\|language=hr\|ep\=5\|lang=hr
language=hr
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_after_$language

ENGLISH_ED_MODEL=model\=global\|language=sl\|ep\=8\|lang=sl
language=sl
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_after_$language

### Train ED model on Croatian, Estonian, Finnish and Slovenian.
for language in 'et' 'fi' 'hr' 'sl'
do
    rm generated/common_top_words_freq_vectors_fasttext.t7
    CUDA_VISIBLE_DEVICES=0 th ed/ed.lua -root_data_dir $DATA_PATH -ent_vecs_filename $ENTITY_VECS -model 'global' -language $language 2>&1 | tee logs/log_only_wikiann_multi-lingual_train_$language
done

### Test the Croatian, Estonian, Finnish and Slovenian datasets with ED model trained on the wikiann datasets.
ENGLISH_ED_MODEL=model\=global\|language=et\|ep\=23\|lang=et
language=et
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_before_$language

ENGLISH_ED_MODEL=model\=global\|language="fi"\|ep\=23\|lang="fi"
language="fi"
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_before_$language

ENGLISH_ED_MODEL=model\=global\|language=hr\|ep\=23\|lang=hr
language=hr
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_before_$language

ENGLISH_ED_MODEL=model\=global\|language=sl\|ep\=8\|lang=sl
language=sl
echo "CUDA_VISIBLE_DEVICES=0 th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL"
th ed/test/test_one_loaded_model.lua -root_data_dir $DATA_PATH -type float -model global -language $language -ent_vecs_filename $ENTITY_VECS -load_ed_model $ENGLISH_ED_MODEL > logs/log_before_$language

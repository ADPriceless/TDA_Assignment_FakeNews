# EDA
## Notes from looking at CSV data
1. "language" column is all English: no value; remove
2. No author in "author" column is spelled in (at least) two ways: "No Author" and "-NO AUTHOR-"
3. There are columns "title" and "text", each of which has a counterpart suffixed by "_without_stopwords"
   - What terms are considered to be *stopwords*?
4. There is a "type" column which categorises the samples into categories such as "bs", "conspiracy", "bias", "satire"
   - These labels may contain biases from the author since some appear to be oppinion
   - These labels may leak information about the labels (e.g. satire and conspiracy are highly likey to be fake since the meaning of these words implies that the articles are fake)
5. There is a "hasImage" column

## Initial Analysis
1. Remove "language" column
2. Check for NaNs
3. Identify and count unique values for columns (to check for columns with different names): author, type 
4. Check correlation of features with labels (e.g. if one authors is a prolific fraudster)
5. Check label imbalance

## Text Data
1. Find out what the stop words are 
2. Investigate NLP libraries (e.g. NLTK, SpaCy, Hugging Face Transformers)
3. Find common words to see what the articles are about: "title" and "text"
4. See if there are any terms that are more common in fake vs. real news articles

## Time Series Data
1. Investigate how fake articles vary with time
   - See if there are spikes around real-world events: using events found from text analysis, as well as researching dates where there are more fraudulent items

# Experimenting
## Data Preprocessing
### Categorical Features
1. Reformat duplicates to remove them

### Text Features
1. Use either the text columns with or without the stopwords
2. Experiment with different ways of encoding the text data (e.g. BERT, word2vec)

## Model
1. Experiment with different models
2. If a sequence model is used, should a classifier be trained on the rest of the data. If so, how two use the two outputs to decide on a final result

---
# Research Resources
1. https://blog.paperspace.com/transformers-text-classification/#importing-the-essential-libraries
2. https://huggingface.co/docs/transformers/tasks/sequence_classification
3. https://www.nltk.org/index.html
4. https://spacy.io/
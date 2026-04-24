# Part 0 : Setup
# ---------------------

# load library
library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(tidyverse)
library(tidytext)
library(ggplot2)
library(forcats)
library(tibble)
library(scales)

# loading the texts
text_a <- read_file("texts/A07594__Circle_of_Commerce.txt")
text_b <- read_file("texts/B14801__Free_Trade.txt")

# Read the raw text files into R
texts <- tibble(
  doc_title = c("Circle", "Free"),
  texts = c(text_a, text_b)
)

# identify stopwords

data("stop_words")
custom_stopwords <- tibble(
  word = c(
    "vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
    "beene","thereof","thus" 
  )
)

all_stopwords <- bind_rows(stop_words, custom_stopwords) %>%
  mutate(word = str_to_lower(word)) %>%
  distinct(word)

# basic clean up
texts_clean <- texts %>%
  mutate(
    texts = texts %>%
      str_replace_all("ſ", "s") %>%
      str_replace_all("\\s+", " ")     
  )



# ---------------------
# Part I : Raw Count Sentiment Analysis
# ---------------------

# Step 1: Tokenize and clean the text 

raw_counts <- texts_clean %>%
  unnest_tokens(word, texts) %>%
  mutate(word = str_to_lower(word)) %>%
  filter(str_detect(word, "^[a-z]+$")) %>%
  anti_join(all_stopwords, by = "word") %>%
  count(doc_title, word, sort = TRUE)
raw_counts

# Step 2: Join to Bing dictionary
bing <- get_sentiments("bing")

sentiment_raw_counts <- raw_counts %>%
  inner_join(bing, by = "word")
sentiment_raw_counts

# Step 3: Raw sentiment totals per document in summary table

summary_sentiment_raw_counts <- sentiment_raw_counts %>%
  group_by(doc_title, sentiment) %>%
  summarize(n=sum(n), .groups = "drop")  %>%
  pivot_wider(
    names_from = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  mutate(sentiment_raw_counts = positive - negative)

summary_sentiment_raw_counts

# ---------------------
# Part II : TF-IDF-Weighted Sentiment Analysis
# ---------------------

# Step 1: Compute TF-IDF for words in each document
tfidf_count <- raw_counts %>%
  bind_tf_idf(term = word, document = doc_title, n= n)

# Step 2: Keep only sentiment-bearing words
sentiment_tfidf_counts <- tfidf_count %>%
  inner_join(bing, by = "word")

# Step 3: Compute raw sentiment totals in second summary table 
summary_sentiment_tfidf <- sentiment_tfidf_counts %>%
  group_by(doc_title) %>%
    summarise(
    tfidf_positive = sum(tf_idf[sentiment == "positive"], na.rm = TRUE),
    tfidf_negative = sum(tf_idf[sentiment == "negative"], na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    net_sentiment_tfidf = tfidf_positive - tfidf_negative
  )

summary_sentiment_tfidf 
  
  
# ---------------------
# Part III : Compare Raw v. TF-IDF Sentiment
# ---------------------

# Step 1: Create final table and export to csv
sentiment_comparison <- summary_sentiment_raw_counts %>%
  left_join(summary_sentiment_tfidf, by = "doc_title") %>%
  arrange(doc_title)

# Check: should be exactly 2 rows (one per document)
print(sentiment_comparison)

# Export as CSV (upload this file to Canvas as instructed)
write_csv(sentiment_comparison, "sentiment_comparison_table.csv")


# Step 2: Answer questions 

# Q1. What changed between the two methods?

## When we use raw word counts, both texts come out as net positive,
## with "Circle" as slightly more positive than "Free Trade".
## This seems to suggest that positive sentiment words are used
## more often than negative words in both texts in absolute counts.
## This result is likely influenced by text length; if positive words
## just appear more overall, the raw count method would emphasize this.
## Notably, after switching to TF-IDF, the net sentiment becomes negative
## for both texts, with Circle being slightly more negative than Free Trade. 

# Q2. Why did TF-IDF alter the results?

## This is because TF-IDF focuses less on raw counts - which can be
## influenced by sheet text length and repetition - and focuses on the relative
## importance of distinctive terms. Here, the more distinctive
## negative terms there are, the more relative influence ('weight') they have 
## on the sentiment analysis. This shift to a weights-based approach
## appears to drive the shift towards a net negative sentiment. 


# Q3. Which specific words drove the changes?

sentiment_tfidf_inspect <- sentiment_tfidf_counts %>%
  group_by(doc_title) %>%
  arrange(desc(tf_idf), .by_group = TRUE) %>%
  slice_head(n = 10) %>%
  View()

## In "Free Trade", words such as 
## imposition, disorderly, deficient, forbidden, offend, cruelty, 
## and extortion received relatively high TF-IDF scores, 
## increasing the influence of negative sentiment under the weighted approach. 

## In Circle of Commerce, words such as ignorance, impertinent, doubt, imperfect, 
## complaint, danger, and superfluous played a similar role.


# Q4. When would each method be more appropriate?

## The raw-count method is more appropriate when the goal is to
## measure the general tone across documents, especially those 
## of a similar length (and perhaps authorship style as well). 
## In contrast, the TF-IDF method is more appropriate when the goal
## is to identify which words are distinctive across texts,
## especially when dealing with texts that are not of the same length,
## or where the styles are different (e.g., more repetition in one over
## the other)

# ---------------------





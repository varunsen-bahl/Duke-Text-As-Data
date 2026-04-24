# ============================================================
# IDS 570: Data Exploration Exercise
# Varun Sen Bahl 
# ============================================================

# ---------------------
# I - Setup
# ---------------------

## 1: load libraries
library(here)
library(readr)
library(tibble)
library(tools)
library(dplyr)
library(tidyr)
library(stringr)
library(tidyverse)
library(tidytext)
library(ggplot2)
library(forcats)
library(scales)
library(quanteda)
library(quanteda.textstats)
library(udpipe)
library(gt)

## 2: loading directories and texts

### directories
text_files <- here("texts")
outputs <- here("outputs")

### texts
all_files <- list.files(text_files, pattern = "\\.txt$", full.names = TRUE)

### text names from EarlyPrint library
name_map <- c(
  "A06785.txt" = "Malynes_Center",
  "A06786.txt" = "Malynes_Consuetudo",
  "A06788.txt" = "Malynes_Englands",
  "A06789.txt" = "Malynes_FreeTrade",
  "A06790.txt" = "Malynes_StGeorge",
  "A06791.txt" = "Malynes_Treatise",
  "A07594.txt" = "Misselden_Circle",
  "A07886.txt" = "Mun_Discourse",
  "A32827.txt" = "Child_Discourse1",
  "A32828.txt" = "Child_Discourse2",
  "A32829.txt" = "Child_Wool",
  "A32830.txt" = "Child_EastIndia",
  "A32833.txt" = "Child_NewDiscourse",
  "A32836.txt" = "Child_Proposals",
  "A32837.txt" = "Child_AdditionTrade",
  "A32838.txt" = "Child_Supplement",
  "A32839.txt" = "Child_Treatise",
  "A50763.txt" = "Child_Method",
  "A51598.txt" = "Mun_Treasure",
  "A69858.txt" = "Child_Discourse3",
  "A93819.txt" = "Child_StateCase",
  "B14801.txt" = "Misselden_FreeTrade",
  "wealth.txt" = "Smith_Wealth"
)

# 3: Read the raw text files into R
text_tbl <- tibble(
  file = basename(all_files),
  doc_id = name_map[file],
  text = sapply(all_files, read_file, USE.NAMES = FALSE)
)

# ---------------------
# II - Preprocessing 
# ---------------------
# 1: Normalization (with backup)

replacements <- c(
 "ſ" = "s",
 "vpon" = "upon"
)

text_tbl <- text_tbl %>%
  mutate(text_clean = text %>%
      str_replace_all(replacements) %>%
      str_replace_all("2dly", "secondly") %>%
      str_replace_all("\\s+", " ") %>%  
      str_to_lower()
  )

corp <- corpus(text_tbl, text_field = "text_clean")

# 2: Tokenization

texts_toks <- tokens(
  corp,
  remove_punct = TRUE,
  remove_numbers = TRUE,
  remove_symbols = TRUE
)

texts_toks <- tokens_tolower(texts_toks)

# 3: Basic stopword removal


custom_stop <- c(
  "vnto","haue","doo","hath","bee","ye","thee","hee","shall","hast","doe",
  "beene","thereof","thus", "answ", "arg", "pag", "em", "etc", "th", "o", "l", "ll"
)

texts_toks <- tokens_remove(texts_toks, pattern = c(stopwords("en"), custom_stop))
# ---------------------
# III - TF-IDF: lexical distinctiveness 
# ---------------------

# 1. Construct a document feature matrix 
dfm_mat <- dfm(texts_toks)

### Inspection by raw count 
# view(topfeatures(dfm_mat, 25))

# 2. Compute TF-IDF weights 
dfm_tfidf <- dfm_tfidf(dfm_mat)
dfm_tfidf

# 3. Extract top 15 TF-IDF terms for each document 
top10_tfidf <- dfm_tfidf %>%
  tidy() %>%
  group_by(document) %>%
  slice_max(order_by= count, n=10, with_ties = FALSE) %>%
  ungroup()

top10_tfidf %>%
  gt() %>%
  gtsave(here("outputs", "top10_tfidf.html"))
  
# 4. Visualize top 10 TF-IDF terms for each document 

part1 <- ggplot(
  top10_tfidf, aes(x = count, y = reorder(term, count))
  )+
  geom_col(width = 0.6) +
  facet_wrap(~ document, scales = "free", ncol = 4) +
  labs(
    title = "Most Characteristic Terms by Document (TF–IDF)",
    x = "TF–IDF weight",
    y = NULL
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_text(size = 8),
    panel.spacing = unit(1, "lines")
  )

ggsave(
  filename = here::here("outputs", "tfidf_plot1.png"),
  plot = part1,
  width = 12,
  height = 10,
  dpi = 300
)

# ---------------------
# IV - Pearson correlation: similarity and distance between texts
# ---------------------

# 1. Trimming very rare words from DFM 
dfm_mat_trim <- dfm_trim(dfm_mat, min_termfreq = 5)

# 2. Use the DFM to compute pairwise Pearson correlations

sim_cor <- textstat_simil(
  dfm_mat_trim,
  method = "correlation",
  margin = "documents"
)

### Convert similarity object to a matrix
r_mat <- as.matrix(sim_cor)

# 3. Visualizing results using similarity heatmap 

r_mat <- round(r_mat, 3)

heat_df <- as.data.frame(r_mat) %>%
  rownames_to_column("doc_i") %>% # First document in pair
  pivot_longer(-doc_i, names_to = "doc_j" , values_to = "r") 

part2 <- ggplot(heat_df, aes(x = doc_j, y = doc_i, fill = r)) +
  geom_tile() + # Create colored tiles
  geom_text(aes(label = round(r, 2)), size = 3)+
  coord_fixed() + # Keep tiles square
  scale_fill_gradient2(
    low = "blue" ,
    mid = "white" ,
    high = "red" ,
    midpoint = 0 # Center color scale at 0
  ) +
  labs( title = "Pearson Correlation Between Documents" ,
        x = NULL,
        y = NULL,
        fill = "Correlation"
  ) + theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    panel.grid = element_blank()
  )

ggsave(
  filename = here::here("outputs", "similarity_heatmap.png"),
  plot = part2,
  width = 12,
  height = 10,
  dpi = 300
)

# 4. Interpretive questions:

### Quick look
dim(r_mat)
r_mat[1:5, 1:5]

# 4(a): identifying two most similar document pairs 
most_similar <- heat_df%>%
  arrange(desc(r)) %>%
  slice_head(n=50)
view(most_similar)

# 4(b): identifying two least similar document pairs 
least_similar <- heat_df%>%
  arrange(r) %>%
  slice_head(n=8)
view(least_similar)

# ---------------------
# V - Syntactic complexity profile 
# ---------------------

# 1: Picking two texts (with explanation)

least_texts <- c(
  "Child_StateCase",
  "Malynes_Treatise"
)

texts_syn <- text_tbl %>%
  filter(doc_id %in% least_texts)

# 2: Creating syntactic complexity framework

### Load an English UD (= Universal Dependencies) model once
model_info <- udpipe_download_model(language = "english-ewt")
ud_model <- udpipe_load_model(model_info$file_model)

### annotate texts using UDPipe 

anno_df <- texts_syn %>%
  mutate(
    # Parse each text with the UD parser; set doc_id to our document name
    anno = map2(text_clean, doc_id, ~ udpipe_annotate(ud_model, x = .x, doc_id = .y) %>%
                  as.data.frame())
  ) %>%
  # Keep only parsed annotations, then unnest into rows
  select(anno) %>%
  unnest(anno) %>%
  # Use the UD doc_id as our document label (and drop any duplicates cleanly)
  rename(document = doc_id) %>%
  # Select columns for syntactic analysis
  select(
    document,
    paragraph_id,
    sentence_id,
    token_id,
    token,
    lemma,
    upos,          # part of speech
    feats,         # grammatical features (e.g., verb form)
    head_token_id, # head of dependency relation
    dep_rel        # dependency relation type
  )

anno_df %>% glimpse()

### binary flags

syntax_df <- anno_df %>%
  mutate(
    is_word = upos != "PUNCT", #<--is it a word (and not punctuation?)
    
    
    # Is this an independent clause? finite verbs are proxy for indipendent clauses
    is_clause = (upos %in% c("VERB", "AUX")) &
      str_detect(coalesce(feats, ""), "VerbForm=Fin"),
    
    # Dependent clause? 
    is_dep_clause = dep_rel %in% c(
      "advcl", #adverbial clause 
      "ccomp", # clausal complement
      "xcomp", #open clausal complement
      "acl", #adnomial clause
      "acl:relcl" #relative clause
    ),
    
    # Is this coordination? That is, does it use "and" "or" etc.?
    is_coord = dep_rel %in% c("conj", "cc"),
    
    # Nominal complexity: these relations make noun phrases more complex
    is_complex_nominal = dep_rel %in% c(
      "amod", # adjective modifier ("big cup")
      "nmod", #nominal modifier ("cup of tea")
      "compound", # compound ("lemon tea")
      "appos" #apposition ("tea, my favorite!")
    )
    
  )

syntax_df %>% 
  select(document, token, upos, is_clause, is_dep_clause) %>%
  head(20)

sentence_df <- syntax_df %>%
  filter(is_word) %>%           #count words (not punctuation)
  group_by(document, sentence_id) %>%   #group by document and sentence
  summarise(
    words          = n(),   #number of words per sentence
    clauses        = sum(is_clause), # number of clauses per sentence
    dep_clauses    = sum(is_dep_clause), #number of dependent clauses per sentence
    .groups = "drop"
  )

sentence_df

## 2(a): mean length of sentence 

mls_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    MLS = mean(words), # Average words per sentence
    .groups = "drop"
  )

mls_df 

## 2(b): clauses per sentence 

clausal_density_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    sentences = n(),
    clauses   = sum(clauses),
    C_per_S   = clauses / sentences,
    .groups = "drop"
  )

clausal_density_df


## 2(c): dependent clauses per clause and/or sentence 


subordination_df <- sentence_df %>%
  group_by(document) %>%
  summarise(
    clauses = sum(clauses),
    dep_clauses = sum(dep_clauses),
    sentences = n(),
    DC_per_C = dep_clauses / pmax(clauses, 1),
    DC_per_S = dep_clauses / sentences,
    .groups = "drop"
  )

view(subordination_df)

## 2(d): coordination per clause and/or sentence 

coordination_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    coord_relations = sum(is_coord),
    clauses         = sum(is_clause),
    sentences       = n_distinct(sentence_id),
    Coord_per_C     = coord_relations / pmax(clauses, 1),
    Coord_per_S     = coord_relations / sentences,
    .groups = "drop"
  )

coordination_df




## 2(e): complex nominals per clause and/or sentence 

nominal_df <- syntax_df %>%
  group_by(document) %>%
  summarise(
    complex_nominals = sum(is_complex_nominal),
    clauses          = sum(is_clause),
    sentences        = n_distinct(sentence_id),
    CN_per_C         = complex_nominals / pmax(clauses, 1),
    CN_per_S         = complex_nominals / sentences,
    .groups = "drop"
  )

nominal_df


# 3: Summary table reporting all syntactic measures for both texts 

all_measures <- mls_df %>%  # ← Added mls_df %>%
  left_join(clausal_density_df %>% select(document, C_per_S), by = "document") %>%
  left_join(subordination_df %>% select(document, DC_per_C, DC_per_S), by = "document") %>%
  left_join(coordination_df %>% select(document, Coord_per_C, Coord_per_S), by = "document") %>%
  left_join(nominal_df %>% select(document, CN_per_C, CN_per_S), by = "document")

all_measures %>%
  knitr::kable(
    digits = 2,
    col.names = c("Document", "MLS", "C/S", "DC/C", "DC/S", 
                  "Coord/C", "Coord/S", "CN/C", "CN/S")
  )

all_measures %>%
  gt() %>%
  gtsave(here("outputs", "syntactic_complexity_table.html"))

### Visualizing the table 

syntax_long <- all_measures %>%  # ← Added %>%
  pivot_longer(
    cols = -document,
    names_to = "Measure",
    values_to = "Value"
  ) %>%
  mutate(
    Category = case_when(
      Measure == "MLS" ~ "Sentence Length",
      Measure == "C_per_S" ~ "Clausal Density",
      Measure %in% c("DC_per_C", "DC_per_S") ~ "Subordination",
      Measure %in% c("Coord_per_C", "Coord_per_S") ~ "Coordination",
      Measure %in% c("CN_per_C", "CN_per_S") ~ "Phrasal Complexity"
    )
  )

# Plot
part3 <- ggplot(syntax_long, aes(x = Measure, y = Value, fill = document)) +
  geom_col(position = "dodge", width = 0.7) +
  facet_wrap(~Category, scales = "free", ncol = 2) +
  scale_fill_brewer(palette = "Set2") +
  labs(
    title = "Syntactic Complexity: Complete Profile",
    subtitle = "Comparing multiple dimensions of syntactic complexity",
    x = NULL,
    y = "Value",
    fill = "Document"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  )

ggsave(
  filename = here::here("outputs", "syntactic_complexity.png"),
  plot = part3,
  width = 12,
  height = 10,
  dpi = 300
)


### Identifying sentences

view(all_measures)




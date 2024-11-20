# Text corpus
The text corpus follows the JSON Lines (JSONL) format, where each line is a separate JSON object representing an individual article with specific fields representing details of the document.
## Required Fields
* PMID: A unique identifier for the article (e.g., PubMed ID).
* ArticleTitle: The title of the article.
* Abstract: The article's abstract.
* MeshHeadingList: A list of Medical Subject Headings (MeSH) terms or other keywords associated with the article.
* full_text: The full text of the article (can be empty if unavailable).
* ArticleType: The type of document (e.g., Original Contribution, Clinical Case Report,  Review Article, etc.)
## Optional Fields
* Journal: The journal in which the article was published.
* PubDate: An object with details of the publication date.
* Year: The publication year.
* Month: The publication month.
* Day: The publication day.
* Season: The publication season (if applicable).
* MedlineDate: Additional date details in Medline format (optional).
* Country: The country of publication.

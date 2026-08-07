--generating embeddings (text in a list of fixed length), so that similar texts are closer to each other.we get this using the sentence transformers
--the angle between the two vectors decide the similarity between two text which is measured by taking the dot product of two vectors and getting their cosine value ie the cosine similarity.

--clustering is also an essential part which is important to cluster the embeddings using kmeans and on top of this add cosine similarity , which makes it eaier to look at the relevant cluster instead running cosine similarity on every emedding.

-- Searched these learning logs ,embedding these and using semantic search, but the results I got were not upto the mark   and the cause turned out to be not having docs chunked as my docs dont just contain one topic.
OK so chunking got much muuch better results .....
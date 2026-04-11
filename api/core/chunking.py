def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50
    ) -> list[str]:
    '''
    Split text into overlapping chunks.
    
    Args:
        text (str): The input text to be chunked.
        chunk_size (int): Number of words per chunk.
        overlap (int): Number of words to repeat between chunks.
    
    Returns:
        List[str]: A list of text chunks.
    
    '''
    
    words = text.split()
    chunks = []
    start = 0
    
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks = chunks.append(chunk)
        start += chunk_size - overlap
        
    return chunks
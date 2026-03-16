FROM eclipse-temurin:21-jre-jammy

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python and required system libraries
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    wget \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir pyserini

# Verify pyserini works at BUILD time
RUN python -c "from pyserini.search.lucene import LuceneSearcher; print('Pyserini OK')"

CMD ["/bin/bash"]
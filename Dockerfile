FROM python:3.10-slim
RUN apt-get update && apt-get install -y build-essential python3-dev git
ARG DISABLE_PREFETCH=0
RUN pip install numpy
RUN git clone --depth 1 https://github.com/nmslib/hnswlib.git /tmp/hnswlib
RUN if [ "$DISABLE_PREFETCH" = "1" ]; then       python3 -c "from pathlib import Path; p = Path('/tmp/hnswlib/hnswlib/hnswalg.h'); t = p.read_text(); marker = '#include <memory>'; inject = chr(10) + '#ifdef DISABLE_HNSW_PREFETCH' + chr(10) + '#define _mm_prefetch(a, sel) ((void)0)' + chr(10) + '#endif' + chr(10); assert marker in t, 'Marker not found in hnswalg.h'; t = t.replace(marker, marker + inject, 1); assert 'DISABLE_HNSW_PREFETCH' in t, 'Patch injection failed'; p.write_text(t)";     fi
RUN if [ "$DISABLE_PREFETCH" = "1" ]; then       CXXFLAGS='-DDISABLE_HNSW_PREFETCH' pip install /tmp/hnswlib;     else       pip install /tmp/hnswlib;     fi && rm -rf /tmp/hnswlib
COPY worker.py /app/worker.py
COPY real_world_dataset.npy /app/real_world_dataset.npy
WORKDIR /app
CMD ["python", "worker.py"]

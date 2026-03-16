# The pre-baked PyLucene
FROM coady/pylucene


WORKDIR /app


#Add your project's specific requirements on top of it
COPY requirements.txt /workspace/
RUN pip install --no-cache-dir -r /workspace/requirements.txt

CMD ["/bin/bash"]
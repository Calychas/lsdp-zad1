ARG VERSION=1.19.3
FROM nginx:$VERSION

RUN apt-get install -y curl
ENV name="/home"

COPY README.md $name

CMD ["Kacper"]
ENTRYPOINT ["echo", "Hello"]

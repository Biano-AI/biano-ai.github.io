# Source codes for Biano AI public website/blog

This repository contains the source code for the Biano AI public blog (it is a simple static page created by [Jekyll](https://jekyllrb.com/)).
The site is automatically published using [Github Pages](https://docs.github.com/en/pages) at [https://biano-ai.github.io/]() .

## Development locally

Run the [Jekyll Docker image](https://github.com/envygeeks/jekyll-docker/blob/master/README.md) in the root of the project directory:

```bash
docker run --rm \
    --volume="$(pwd):/srv/jekyll" \
    -p 127.0.0.1:8080:4000/tcp \
    -it \
    jekyll/jekyll:latest jekyll serve --watch --force_polling --incremental --livereload
```

Now open in a browser: [http://localhost:8080/]().

:point_right: Note: Automatic reload after a change doesn't seem to work on Windows.
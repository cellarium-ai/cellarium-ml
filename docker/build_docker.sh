#!/bin/bash

docker build -t cellarium-ml:latest --build-arg="GIT_SHA=751929262555d7bf8c141cc693c003bc26dd2d1f" .

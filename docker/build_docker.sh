#!/bin/bash

docker build -t cellarium-ml:latest --build-arg="GIT_SHA=b887dd6184d8c08896d396b14ca9f2a469ad245f" .

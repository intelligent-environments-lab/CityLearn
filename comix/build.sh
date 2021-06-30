#!/bin/bash

echo 'Building Dockerfile with image name pymarl'
docker build --build-arg UID=$UID -t pymarl .

#!/bin/bash

TRAILING_VERSION="0.7.0"
LATEST_VERSION="0.7.1"
for version in "0.0.2" "0.1.0" "0.2.0" "0.3.0" "0.4.0" "0.5.0" "0.5.1" "0.6.0"
do
    ln -s vectorscale--${TRAILING_VERSION}--${LATEST_VERSION}.sql vectorscale--${version}--${LATEST_VERSION}.sql
done

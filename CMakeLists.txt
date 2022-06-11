# Specify the minimum cmake version
cmake_minimum_required(VERSION 3.15)

# Define project name and language
project(mpi3tests LANGUAGES CXX)

# Add executables
add_executable(mpi3tests test1.cxx)
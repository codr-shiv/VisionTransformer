set -e  # this guy is a natural ctrl c :D (exit if cmnd fails)

SRC_DIR=src 
INCLUDE_DIR=include
BUILD_DIR=build
MAIN_FILE=main.c
CC=gcc
CFLAGS="-I$INCLUDE_DIR -Wall -O2"

LIB_NAME=ViT

mkdir -p $BUILD_DIR

echo "=== Compiling library files ==="
LIB_OBJS=()

count=0
for src_file in $SRC_DIR/*.c; do
    if [ "$src_file" -nt "main" ]; then
        obj_file="$BUILD_DIR/$(basename ${src_file%.c}.o)"
        echo "    $((++count)) Compiling $src_file -> $obj_file"
        $CC $CFLAGS -c "$src_file" -o "$obj_file" -lm -O3 -march=native -fopenmp
        LIB_OBJS+=("$obj_file")
    fi
done

echo "    Creating static library lib$LIB_NAME.a"
ar rcs "$BUILD_DIR/lib$LIB_NAME.a" "${LIB_OBJS[@]}"

echo "=== Compiling main program ==="
MAIN_OBJ="$BUILD_DIR/main.o"
$CC $CFLAGS -c "$MAIN_FILE" -o "$MAIN_OBJ" -lm -O3 -march=native -fopenmp

echo "=== Linking executable ==="
$CC "$MAIN_OBJ" -L$BUILD_DIR -l$LIB_NAME -o main -lm -fopenmp -march=native -fopenmp

# mavx2 = lets compiler use 256 bit vector operations, faster math if auto vectorized (it generally is)
# mfma = a*b + c in one instruction isntead of 2, faster
# march=native = optimize as per ur cpu (the top 2 automatically)
# O3 =  high optimization, can reorder loops and shi
# lm = link math lib
# fopenmp = enables multithreading

echo "Build complete! Executable: main"

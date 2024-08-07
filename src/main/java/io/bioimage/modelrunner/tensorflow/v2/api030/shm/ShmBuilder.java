/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer API for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *      http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v2.api030.shm;

import io.bioimage.modelrunner.system.PlatformDetection;
import io.bioimage.modelrunner.tensor.shm.SharedMemoryArray;
import io.bioimage.modelrunner.utils.CommonUtils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Arrays;

import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

/**
 * A {@link RandomAccessibleInterval} builder for TensorFlow {@link Tensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Tensorflow 2 {@link Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ShmBuilder
{
    /**
     * Utility class.
     */
    private ShmBuilder()
    {
    }

	/**
	 * Creates a {@link RandomAccessibleInterval} from a given {@link TType} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link TType} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the {@link TType} tensor.
	 * @throws IllegalArgumentException If the {@link TType} tensor type is not supported.
	 * @throws IOException 
	 */
    public static void build(TType tensor, String memoryName) throws IllegalArgumentException, IOException
    {
    	if (tensor instanceof TUint8)
        {
            buildFromTensorUByte((TUint8) tensor, memoryName);
        }
        else if (tensor instanceof TInt32)
        {
            buildFromTensorInt((TInt32) tensor, memoryName);
        }
        else if (tensor instanceof TFloat32)
        {
            buildFromTensorFloat((TFloat32) tensor, memoryName);
        }
        else if (tensor instanceof TFloat64)
        {
            buildFromTensorDouble((TFloat64) tensor, memoryName);
        }
        else if (tensor instanceof TInt64)
        {
            buildFromTensorLong((TInt64) tensor, memoryName);
        }
        else
        {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned byte-typed {@link TUint8} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TUint8} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link UnsignedByteType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorUByte(TUint8 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 1;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.asRawTensor().data().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int32-typed {@link TInt32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link IntType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorInt(TInt32 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 4;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.asRawTensor().data().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float32-typed {@link TFloat32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat32} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link FloatType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorFloat(TFloat32 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 4;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.asRawTensor().data().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned float64-typed {@link TFloat64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link DoubleType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorDouble(TFloat64 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 8;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.asRawTensor().data().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }

	/**
	 * Builds a {@link RandomAccessibleInterval} from a unsigned int64-typed {@link TInt64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt64} tensor data is read from.
	 * @return The {@link RandomAccessibleInterval} built from the tensor, of type {@link LongType}.
	 * @throws IOException 
	 */
    private static void buildFromTensorLong(TInt64 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), false, true);
        ByteBuffer buff = shma.getDataBuffer();
        int totalSize = 8;
		for (long i : arrayShape) {totalSize *= i;}
    	byte[] flatArr = new byte[buff.capacity()];
    	buff.get(flatArr);
        tensor.asRawTensor().data().read(flatArr, flatArr.length - totalSize, totalSize);
        shma.setBuffer(ByteBuffer.wrap(flatArr));
        if (PlatformDetection.isWindows()) shma.close();
    }
}

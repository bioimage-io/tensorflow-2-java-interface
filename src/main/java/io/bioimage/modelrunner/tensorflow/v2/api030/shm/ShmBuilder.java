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
 * A utility class that converts {@link TType} tensors into {@link SharedMemoryArray}s for
 * interprocessing communication
 * 
 * @author Carlos Garcia Lopez de Haro
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
     * Create a {@link SharedMemoryArray} from a {@link TType} tensor
     * @param tensor
     * 	the tensor to be passed into the other process through the shared memory
     * @param memoryName
     * 	the name of the memory region where the tensor is going to be copied
     * @throws IllegalArgumentException if the data type of the tensor is not supported
     * @throws IOException if there is any error creating the shared memory array
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

    private static void buildFromTensorUByte(TUint8 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 1))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per ubyte output tensor supported: " + Integer.MAX_VALUE / 1);
        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new UnsignedByteType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.asRawTensor().data().read(buff.array(), 0, buff.capacity());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorInt(TInt32 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per int output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new IntType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.asRawTensor().data().read(buff.array(), 0, buff.capacity());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorFloat(TFloat32 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 4))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per float output tensor supported: " + Integer.MAX_VALUE / 4);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new FloatType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.asRawTensor().data().read(buff.array(), 0, buff.capacity());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorDouble(TFloat64 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per double output tensor supported: " + Integer.MAX_VALUE / 8);

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new DoubleType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.asRawTensor().data().read(buff.array(), 0, buff.capacity());
        if (PlatformDetection.isWindows()) shma.close();
    }

    private static void buildFromTensorLong(TInt64 tensor, String memoryName) throws IOException
    {
    	long[] arrayShape = tensor.shape().asArray();
		if (CommonUtils.int32Overflows(arrayShape, 8))
			throw new IllegalArgumentException("Model output tensor with shape " + Arrays.toString(arrayShape) 
					+ " is too big. Max number of elements per long output tensor supported: " + Integer.MAX_VALUE / 8);
		

        SharedMemoryArray shma = SharedMemoryArray.readOrCreate(memoryName, arrayShape, new LongType(), false, true);
        ByteBuffer buff = shma.getDataBufferNoHeader();
        tensor.asRawTensor().data().read(buff.array(), 0, buff.capacity());
        if (PlatformDetection.isWindows()) shma.close();
    }
}

/*-
 * #%L
 * This project complements the DL-model runner acting as the engine that works loading models 
 * 	and making inference with Java 0.3.0 and newer APIs for Tensorflow 2.
 * %%
 * Copyright (C) 2022 - 2023 Institut Pasteur and BioImage.IO developers.
 * %%
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 * #L%
 */
package io.bioimage.modelrunner.tensorflow.v2.api030.tensor;

import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImgFactory;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;

import org.tensorflow.Tensor;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

/**
 * A {@link Img} builder for TensorFlow {@link Tensor} objects.
 * Build ImgLib2 objects (backend of {@link io.bioimage.modelrunner.tensor.Tensor})
 * from Tensorflow 2 {@link Tensor}
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ImgLib2Builder
{
    /**
     * Utility class.
     */
    private ImgLib2Builder()
    {
    }

	/**
	 * Creates a {@link Img} from a given {@link TType} tensor
	 * 
	 * @param <T> 
	 * 	the possible ImgLib2 datatypes of the image
	 * @param tensor 
	 * 	The {@link TType} tensor data is read from.
	 * @return The {@link Img} built from the {@link TType} tensor.
	 * @throws IllegalArgumentException If the {@link TType} tensor type is not supported.
	 */
    public static <T extends Type<T>> Img<T> build(TType tensor) throws IllegalArgumentException
    {
    	if (tensor instanceof TUint8)
        {
            return (Img<T>) buildFromTensorUByte((TUint8) tensor);
        }
        else if (tensor instanceof TInt32)
        {
            return (Img<T>) buildFromTensorInt((TInt32) tensor);
        }
        else if (tensor instanceof TFloat32)
        {
            return (Img<T>) buildFromTensorFloat((TFloat32) tensor);
        }
        else if (tensor instanceof TFloat64)
        {
            return (Img<T>) buildFromTensorDouble((TFloat64) tensor);
        }
        else if (tensor instanceof TInt64)
        {
            return (Img<T>) buildFromTensorLong((TInt64) tensor);
        }
        else
        {
            throw new IllegalArgumentException("Unsupported tensor type: " + tensor.dataType().name());
        }
    }

	/**
	 * Builds a {@link Img} from a unsigned byte-typed {@link TUint8} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TUint8} tensor data is read from.
	 * @return The {@link Img} built from the tensor, of type {@link UnsignedByteType}.
	 */
    private static Img<UnsignedByteType> buildFromTensorUByte(TUint8 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ArrayImgFactory< UnsignedByteType > factory = new ArrayImgFactory<>( new UnsignedByteType() );
        final Img< UnsignedByteType > outputImg = factory.create(tensorShape);
    	Cursor<UnsignedByteType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        byte[] flatArr = new byte[totalSize];
        tensor.asRawTensor().data().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = flatArr[flatPos];
			if (val < 0)
				tensorCursor.get().set(256 + (int) val);
			else
				tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a unsigned int32-typed {@link TInt32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt32} tensor data is read from.
	 * @return The {@link Img} built from the tensor, of type {@link IntType}.
	 */
    private static Img<IntType> buildFromTensorInt(TInt32 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ArrayImgFactory< IntType > factory = new ArrayImgFactory<>( new IntType() );
        final Img< IntType > outputImg = factory.create(tensorShape);
    	Cursor<IntType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        int[] flatArr = new int[totalSize];
        tensor.asRawTensor().data().asInts().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	int val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a unsigned float32-typed {@link TFloat32} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat32} tensor data is read from.
	 * @return The {@link Img} built from the tensor, of type {@link FloatType}.
	 */
    private static Img<FloatType> buildFromTensorFloat(TFloat32 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ArrayImgFactory< FloatType > factory = new ArrayImgFactory<>( new FloatType() );
        final Img< FloatType > outputImg = factory.create(tensorShape);
    	Cursor<FloatType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        float[] flatArr = new float[totalSize];
        tensor.asRawTensor().data().asFloats().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	float val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a unsigned float64-typed {@link TFloat64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TFloat64} tensor data is read from.
	 * @return The {@link Img} built from the tensor, of type {@link DoubleType}.
	 */
    private static Img<DoubleType> buildFromTensorDouble(TFloat64 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ArrayImgFactory< DoubleType > factory = new ArrayImgFactory<>( new DoubleType() );
        final Img< DoubleType > outputImg = factory.create(tensorShape);
    	Cursor<DoubleType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        double[] flatArr = new double[totalSize];
        tensor.asRawTensor().data().asDoubles().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	double val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

	/**
	 * Builds a {@link Img} from a unsigned int64-typed {@link TInt64} tensor.
	 * 
	 * @param tensor 
	 * 	The {@link TInt64} tensor data is read from.
	 * @return The {@link Img} built from the tensor, of type {@link LongType}.
	 */
    private static Img<LongType> buildFromTensorLong(TInt64 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ArrayImgFactory< LongType > factory = new ArrayImgFactory<>( new LongType() );
        final Img< LongType > outputImg = factory.create(tensorShape);
    	Cursor<LongType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        long[] flatArr = new long[totalSize];
        tensor.asRawTensor().data().asLongs().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	long val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }
}

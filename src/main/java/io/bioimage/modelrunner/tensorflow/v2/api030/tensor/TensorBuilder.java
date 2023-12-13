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

package io.bioimage.modelrunner.tensorflow.v2.api030.tensor;

import io.bioimage.modelrunner.tensor.Utils;
import io.bioimage.modelrunner.utils.CommonUtils;
import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.NativeType;
import net.imglib2.type.numeric.RealType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Cast;
import net.imglib2.util.Util;
import net.imglib2.view.Views;

import java.util.Arrays;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.buffer.raw.RawDataBufferFactory;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

/**
 * A TensorFlow 2 {@link Tensor} builder from {@link Img} and
 * {@link io.bioimage.modelrunner.tensor.Tensor} objects.
 * 
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link TType} instance with the same size and information as the
	 * given {@link io.bioimage.modelrunner.tensor.Tensor}.
	 * 
	 * @param tensor 
	 * 	The dlmodel-runner {@link io.bioimage.modelrunner.tensor.Tensor} that is
	 * 	going to be converted into a {@link TType} tensor
	 * @return The created {@link TType} tensor.
	 * @throws IllegalArgumentException If the type of the {@link io.bioimage.modelrunner.tensor.Tensor}
	 * is not supported
	 */
	public static TType build(io.bioimage.modelrunner.tensor.Tensor tensor)
		throws IllegalArgumentException
	{
		return build(tensor.getData());
	}

	/**
	 * Creates {@link TType} instance with the same size and information as the
	 * given {@link RandomAccessibleInterval}.
	 * 
	 * @param <T>
	 * 	the ImgLib2 data types the {@link RandomAccessibleInterval} can be
	 * @param array
	 * 	the {@link RandomAccessibleInterval} that is going to be converted into
	 *  a {@link TType} tensor
	 * @return a {@link TType} tensor
	 * @throws IllegalArgumentException if the type of the {@link RandomAccessibleInterval}
	 *  is not supported
	 */
	public static <T extends RealType<T> & NativeType<T>> TType build(
		RandomAccessibleInterval<T> array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (Util.getTypeFromInterval(array) instanceof UnsignedByteType) {
			return buildUByte(Cast.unchecked(array));
		}
		else if (Util.getTypeFromInterval(array) instanceof IntType) {
			return buildInt(Cast.unchecked(array));
		}
		else if (Util.getTypeFromInterval(array) instanceof FloatType) {
			return buildFloat(Cast.unchecked(array));
		}
		else if (Util.getTypeFromInterval(array) instanceof DoubleType) {
			return buildDouble(Cast.unchecked(array));
		}
		else if (Util.getTypeFromInterval(array) instanceof LongType) {
			return buildLong(Cast.unchecked(array));
		}
		else {
			throw new IllegalArgumentException("Unsupported tensor type: " + Util
				.getTypeFromInterval(array).getClass().toString());
		}
	}

	/**
	 * Creates a {@link TType} tensor of type {@link TUint8} from an
	 * {@link RandomAccessibleInterval} of type {@link UnsignedByteType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TType} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TUint8 buildUByte(RandomAccessibleInterval<UnsignedByteType> tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final byte[] flatArr = new byte[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<UnsignedByteType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().getByte();
		}
		ByteDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TUint8 ndarray = Tensor.of(TUint8.class, Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	/**
	 * Creates a {@link TInt32} tensor of type {@link TInt32} from an
	 * {@link RandomAccessibleInterval} of type {@link IntType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TInt32} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TInt32 buildInt(RandomAccessibleInterval<IntType> tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final int[] flatArr = new int[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<IntType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		IntDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TInt32 ndarray = TInt32.tensorOf(Shape.of(ogShape),
			dataBuffer);
		return ndarray;
	}

	/**
	 * Creates a {@link TInt64} tensor of type {@link TInt64} from an
	 * {@link RandomAccessibleInterval} of type {@link LongType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TInt64} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	private static TInt64 buildLong(RandomAccessibleInterval<LongType> tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final long[] flatArr = new long[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<LongType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		LongDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TInt64 ndarray = TInt64.tensorOf(Shape.of(ogShape),
			dataBuffer);
		return ndarray;
	}

	/**
	 * Creates a {@link TFloat32} tensor of type {@link TFloat32} from an
	 * {@link RandomAccessibleInterval} of type {@link FloatType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TFloat32} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TFloat32 buildFloat(
		RandomAccessibleInterval<FloatType> tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final float[] flatArr = new float[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<FloatType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		FloatDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TFloat32 ndarray = TFloat32.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}

	/**
	 * Creates a {@link TFloat64} tensor of type {@link TFloat64} from an
	 * {@link RandomAccessibleInterval} of type {@link DoubleType}
	 * 
	 * @param tensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TFloat64} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	private static TFloat64 buildDouble(
		RandomAccessibleInterval<DoubleType> tensor)
		throws IllegalArgumentException
	{
		long[] ogShape = tensor.dimensionsAsLongArray();
		if (CommonUtils.int32Overflows(ogShape))
			throw new IllegalArgumentException("Provided tensor with shape " + Arrays.toString(ogShape) 
								+ " is too big. Max number of elements per tensor supported: " + Integer.MAX_VALUE);
		tensor = Utils.transpose(tensor);
		long[] tensorShape = tensor.dimensionsAsLongArray();
		int size = 1;
		for (long ll : tensorShape) size *= ll;
		final double[] flatArr = new double[size];
		int[] sArr = new int[tensorShape.length];
		for (int i = 0; i < sArr.length; i ++)
			sArr[i] = (int) tensorShape[i];

		Cursor<DoubleType> cursor = Views.flatIterable(tensor).cursor();
		int i = 0;
		while (cursor.hasNext()) {
			cursor.fwd();
			flatArr[i ++] = cursor.get().get();
		}
		DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TFloat64 ndarray = TFloat64.tensorOf(Shape.of(ogShape), dataBuffer);
		return ndarray;
	}
}

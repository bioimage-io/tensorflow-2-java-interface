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

import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.integer.UnsignedByteType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;
import net.imglib2.view.IntervalView;

import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.buffer.raw.RawDataBufferFactory;
import org.tensorflow.proto.framework.DataType;
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
	public static <T extends Type<T>> TType build(
		RandomAccessibleInterval<T> array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (Util.getTypeFromInterval(array) instanceof UnsignedByteType) {
			return buildUByte((RandomAccessibleInterval<UnsignedByteType>) array);
		}
		else if (Util.getTypeFromInterval(array) instanceof IntType) {
			return buildInt((RandomAccessibleInterval<IntType>) array);
		}
		else if (Util.getTypeFromInterval(array) instanceof FloatType) {
			return buildFloat((RandomAccessibleInterval<FloatType>) array);
		}
		else if (Util.getTypeFromInterval(array) instanceof DoubleType) {
			return buildDouble((RandomAccessibleInterval<DoubleType>) array);
		}
		else if (Util.getTypeFromInterval(array) instanceof LongType) {
			return buildLong((RandomAccessibleInterval<LongType>) array);
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
	 * @param imgTensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TType} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TUint8 buildUByte(RandomAccessibleInterval<UnsignedByteType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<UnsignedByteType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<UnsignedByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor =
			((Img<UnsignedByteType>) imgTensor).cursor();
		else throw new IllegalArgumentException("The data of the " + Tensor.class +
			" has " + "to be an instance of " + Img.class + " or " +
			IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) {
			flatSize *= dd;
		}
		byte[] flatArr = new byte[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			byte val = tensorCursor.get().getByte();
			flatArr[flatPos] = val;
		}
		ByteDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TUint8 tensor = Tensor.of(TUint8.class, Shape.of(imgTensor
			.dimensionsAsLongArray()), dataBuffer);
		return tensor;
	}

	/**
	 * Creates a {@link TInt32} tensor of type {@link TInt32} from an
	 * {@link RandomAccessibleInterval} of type {@link IntType}
	 * 
	 * @param imgTensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TInt32} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TInt32 buildInt(RandomAccessibleInterval<IntType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<IntType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<IntType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor = ((Img<IntType>) imgTensor)
			.cursor();
		else throw new IllegalArgumentException("The data of the " + Tensor.class +
			" has " + "to be an instance of " + Img.class + " or " +
			IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) {
			flatSize *= dd;
		}
		int[] flatArr = new int[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			int val = tensorCursor.get().getInt();
			flatArr[flatPos] = val;
		}
		IntDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TInt32 tensor = TInt32.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()),
			dataBuffer);
		return tensor;
	}

	/**
	 * Creates a {@link TInt64} tensor of type {@link TInt64} from an
	 * {@link RandomAccessibleInterval} of type {@link LongType}
	 * 
	 * @param imgTensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TInt64} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	private static TInt64 buildLong(RandomAccessibleInterval<LongType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<LongType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<LongType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor =
			((Img<LongType>) imgTensor).cursor();
		else throw new IllegalArgumentException("The data of the " + Tensor.class +
			" has " + "to be an instance of " + Img.class + " or " +
			IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) {
			flatSize *= dd;
		}
		long[] flatArr = new long[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			long val = tensorCursor.get().getLong();
			flatArr[flatPos] = val;
		}
		LongDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TInt64 tensor = TInt64.tensorOf(Shape.of(imgTensor.dimensionsAsLongArray()),
			dataBuffer);
		return tensor;
	}

	/**
	 * Creates a {@link TFloat32} tensor of type {@link TFloat32} from an
	 * {@link RandomAccessibleInterval} of type {@link FloatType}
	 * 
	 * @param imgTensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TFloat32} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	public static TFloat32 buildFloat(
		RandomAccessibleInterval<FloatType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<FloatType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<FloatType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor =
			((Img<FloatType>) imgTensor).cursor();
		else throw new IllegalArgumentException("The data of the " + Tensor.class +
			" has " + "to be an instance of " + Img.class + " or " +
			IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) {
			flatSize *= dd;
		}
		float[] flatArr = new float[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			float val = tensorCursor.get().getRealFloat();
			flatArr[flatPos] = val;
		}
		FloatDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TFloat32 tensor = TFloat32.tensorOf(Shape.of(imgTensor
			.dimensionsAsLongArray()), dataBuffer);
		return tensor;
	}

	/**
	 * Creates a {@link TFloat64} tensor of type {@link TFloat64} from an
	 * {@link RandomAccessibleInterval} of type {@link DoubleType}
	 * 
	 * @param imgTensor 
	 * 	The {@link RandomAccessibleInterval} to fill the tensor with.
	 * @return The {@link TFloat64} tensor filled with the {@link RandomAccessibleInterval} data.
	 * @throws IllegalArgumentException if the input {@link RandomAccessibleInterval} type is
	 * not compatible
	 */
	private static TFloat64 buildDouble(
		RandomAccessibleInterval<DoubleType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<DoubleType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<DoubleType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor =
			((Img<DoubleType>) imgTensor).cursor();
		else throw new IllegalArgumentException("The data of the " + Tensor.class +
			" has " + "to be an instance of " + Img.class + " or " +
			IntervalView.class);
		long flatSize = 1;
		for (long dd : imgTensor.dimensionsAsLongArray()) {
			flatSize *= dd;
		}
		double[] flatArr = new double[(int) flatSize];
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
			int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos,
				tensorShape);
			double val = tensorCursor.get().getRealDouble();
			flatArr[flatPos] = val;
		}
		DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(flatArr, false);
		TFloat64 tensor = TFloat64.tensorOf(Shape.of(imgTensor
			.dimensionsAsLongArray()), dataBuffer);
		return tensor;
	}
}

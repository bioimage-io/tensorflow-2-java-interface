
package io.bioimage.modelrunner.tensorflow.v2.api030.tensor;

import io.bioimage.modelrunner.utils.IndexingUtils;

import net.imglib2.Cursor;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
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
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class TensorBuilder {

	/**
	 * Utility class.
	 */
	private TensorBuilder() {}

	/**
	 * Creates {@link TType} instance with the same size and information as the
	 * given {@link RandomAccessibleInterval}.
	 * 
	 * @param tensor The sequence which the created tensor is filled with.
	 * @return The created tensor.
	 * @throws IllegalArgumentException If the type of the sequence is not
	 *           supported.
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
	 * @param array The sequence which the created tensor is filled with.
	 * @return The created tensor.
	 * @throws IllegalArgumentException If the type of the sequence is not
	 *           supported.
	 */
	public static <T extends Type<T>> TType build(
		RandomAccessibleInterval<T> array) throws IllegalArgumentException
	{
		// Create an Icy sequence of the same type of the tensor
		if (Util.getTypeFromInterval(array) instanceof ByteType) {
			return buildByte((RandomAccessibleInterval<ByteType>) array);
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
	 * {@link RandomAccessibleInterval} of type {@link DataType#DT_INT8} or
	 * {@link DataType#DT_UINT8}.
	 * 
	 * @param imgTensor The sequence to fill the tensor with.
	 * @return The tensor filled with the image data.
	 * @throws IllegalArgumentException If the type of the image is not
	 *           compatible.
	 */
	public static TUint8 buildByte(RandomAccessibleInterval<ByteType> imgTensor)
		throws IllegalArgumentException
	{
		long[] tensorShape = imgTensor.dimensionsAsLongArray();
		Cursor<ByteType> tensorCursor;
		if (imgTensor instanceof IntervalView) tensorCursor =
			((IntervalView<ByteType>) imgTensor).cursor();
		else if (imgTensor instanceof Img) tensorCursor =
			((Img<ByteType>) imgTensor).cursor();
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
	 * Creates a {@link TType} tensor of type {@link TInt32} from an
	 * {@link RandomAccessibleInterval} of type {@link DataType#DT_INT16} or
	 * {@link DataType#DT_UINT16}.
	 * 
	 * @param imgTensor The sequence to fill the tensor with.
	 * @return The tensor filled with the image data.
	 * @throws IllegalArgumentException If the type of the image is not
	 *           compatible.
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
	 * Creates a {@link Tensor} of type {@link TInt64} from an
	 * {@link RandomAccessibleInterval} of type {@link DataType#DT_INT64}
	 * 
	 * @param imgTensor The sequence to fill the tensor with.
	 * @return The tensor filled with the image data.
	 * @throws IllegalArgumentException If the type of the image is not
	 *           compatible.
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
	 * Creates a {@link TType} tensor of type {@link TFloat32} from an
	 * {@link RandomAccessibleInterval} of type {@link DataType#DT_FLOAT}.
	 * 
	 * @param imgTensor The sequence to fill the tensor with.
	 * @return The tensor filled with the image data.
	 * @throws IllegalArgumentException If the type of the image is not
	 *           compatible.
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
	 * Creates a {@link Tensor} of type {@link TFloat64} from an
	 * {@link RandomAccessibleInterval} of type {@link DataType#DT_DOUBLE}.
	 * 
	 * @param imgTensor The sequence to fill the tensor with.
	 * @return The tensor filled with the image data.
	 * @throws IllegalArgumentException If the type of the image is not
	 *           compatible.
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

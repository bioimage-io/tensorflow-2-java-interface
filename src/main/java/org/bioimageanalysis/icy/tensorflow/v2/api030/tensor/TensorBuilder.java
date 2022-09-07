package org.bioimageanalysis.icy.tensorflow.v2.api030.tensor;

import org.bioimageanalysis.icy.deeplearning.tensor.RaiArrayUtils;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.buffer.ByteDataBuffer;
import org.tensorflow.ndarray.buffer.DoubleDataBuffer;
import org.tensorflow.ndarray.buffer.FloatDataBuffer;
import org.tensorflow.ndarray.buffer.IntDataBuffer;
import org.tensorflow.ndarray.buffer.LongDataBuffer;
import org.tensorflow.ndarray.impl.buffer.raw.RawDataBufferFactory;
import org.tensorflow.types.TBool;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.RandomAccessibleInterval;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;
import net.imglib2.util.Util;


/**
 * @author Carlos Garcia Lopez de Haro and Daniel Felipe Gonzalez Obando 
 */
public final class TensorBuilder
{
    /**
     * Utility class.
     */
    private TensorBuilder()
    {
    }

    /**
     * Creates {@link TType} instance with the same size and information as the given {@link RandomAccessibleInterval}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static TType build(org.bioimageanalysis.icy.deeplearning.tensor.Tensor tensor) throws IllegalArgumentException
    {
    	return build(tensor.getData());
    }

    /**
     * Creates {@link TType} instance with the same size and information as the given {@link RandomAccessibleInterval}.
     * 
     * @param sequence
     *        The sequence which the created tensor is filled with.
     * @return The created tensor.
     * @throws IllegalArgumentException
     *         If the type of the sequence is not supported.
     */
    public static <T extends Type<T>> TType build(RandomAccessibleInterval<T> array) throws IllegalArgumentException
    {
        // Create an Icy sequence of the same type of the tensor
    	if (Util.getTypeFromInterval(array) instanceof ByteType) {
            return buildByte((RandomAccessibleInterval<ByteType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof IntType) {
            return buildInt((RandomAccessibleInterval<IntType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof FloatType) {
            return buildFloat((RandomAccessibleInterval<FloatType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof DoubleType) {
            return buildDouble((RandomAccessibleInterval<DoubleType>) array);
    	} else if (Util.getTypeFromInterval(array) instanceof LongType) {
            return buildLong((RandomAccessibleInterval<LongType>) array);
    	} else {
            throw new IllegalArgumentException("Unsupported tensor type: " + Util.getTypeFromInterval(array).getClass().toString());
    	}
    }

    /**
     * Creates a {@link TType} tensor of type {@link TUint8} from an {@link RandomAccessibleInterval} of type {@link DataType#BYTE} or {@link DataType#UBYTE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TUint8 buildByte(RandomAccessibleInterval<ByteType> array) throws IllegalArgumentException
    {
        ByteDataBuffer dataBuffer = RawDataBufferFactory.create(RaiArrayUtils.byteArray(array), false);
        TUint8 tensor = Tensor.of(TUint8.class, Shape.of(array.dimensionsAsLongArray()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link TType} tensor of type {@link TInt32} from an {@link RandomAccessibleInterval} of type 
     * {@link DataType#INT} or
     * {@link DataType#UINT}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TInt32 buildInt(RandomAccessibleInterval<IntType> array) throws IllegalArgumentException
    {
        IntDataBuffer dataBuffer = RawDataBufferFactory.create(RaiArrayUtils.intArray(array), false);
        TInt32 tensor = TInt32.tensorOf(Shape.of(array.dimensionsAsLongArray()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TInt64} from an {@link RandomAccessibleInterval} of type {@link DataType#LONG}
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static TInt64 buildLong(RandomAccessibleInterval<LongType> array) throws IllegalArgumentException
    {
        LongDataBuffer dataBuffer = RawDataBufferFactory.create(RaiArrayUtils.longArray(array), false);
        TInt64 tensor = TInt64.tensorOf(Shape.of(array.dimensionsAsLongArray()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link TType} tensor of type {@link TFloat32} from an {@link RandomAccessibleInterval} of type {@link DataType#FLOAT}.
     * 
     * @param sequence
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    public static TFloat32 buildFloat(RandomAccessibleInterval<FloatType> array) throws IllegalArgumentException
    {
        FloatDataBuffer dataBuffer = RawDataBufferFactory.create(RaiArrayUtils.floatArray(array), false);
        TFloat32 tensor = TFloat32.tensorOf(Shape.of(array.dimensionsAsLongArray()), dataBuffer);
		return tensor;
    }

    /**
     * Creates a {@link Tensor} of type {@link TFloat64} from an {@link RandomAccessibleInterval} of type {@link DataType#DOUBLE}.
     * 
     * @param array
     *        The sequence to fill the tensor with.
     * @return The tensor filled with the image data.
     * @throws IllegalArgumentException
     *         If the type of the image is not compatible.
     */
    private static TFloat64 buildDouble(RandomAccessibleInterval<DoubleType> array) throws IllegalArgumentException
    {
        DoubleDataBuffer dataBuffer = RawDataBufferFactory.create(RaiArrayUtils.doubleArray(array), false);
        TFloat64 tensor = TFloat64.tensorOf(Shape.of(array.dimensionsAsLongArray()), dataBuffer);
		return tensor;
    }
}

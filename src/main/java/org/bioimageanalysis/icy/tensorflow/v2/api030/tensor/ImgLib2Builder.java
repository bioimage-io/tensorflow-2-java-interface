package org.bioimageanalysis.icy.tensorflow.v2.api030.tensor;

import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.img.Img;
import net.imglib2.img.array.ArrayImg;
import net.imglib2.img.array.ArrayImgs;
import net.imglib2.img.basictypeaccess.array.ByteArray;
import net.imglib2.img.basictypeaccess.array.DoubleArray;
import net.imglib2.img.basictypeaccess.array.FloatArray;
import net.imglib2.img.basictypeaccess.array.IntArray;
import net.imglib2.img.basictypeaccess.array.LongArray;
import net.imglib2.type.Type;
import net.imglib2.type.numeric.integer.ByteType;
import net.imglib2.type.numeric.integer.IntType;
import net.imglib2.type.numeric.integer.LongType;
import net.imglib2.type.numeric.real.DoubleType;
import net.imglib2.type.numeric.real.FloatType;


/**
 * @author Carlos GArcia Lopez de Haro and Daniel Felipe Gonzalez Obando
 */
public final class ImgLib2Builder
{
    /**
     * Utility class.
     */
    private ImgLib2Builder()
    {
    }

    public static <T extends Type<T>> Img<T> build(TType tensor) throws IllegalArgumentException
    {
    	if (tensor instanceof TUint8)
        {
            return (Img<T>) buildFromTensorByte((TUint8) tensor);
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

    private static ArrayImg<ByteType, ByteArray> buildFromTensorByte(TUint8 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		byte[] flatImageArray = new byte[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().read(flatImageArray);
		return ArrayImgs.bytes(flatImageArray, tensorShape);
    }

    private static ArrayImg<IntType, IntArray> buildFromTensorInt(TInt32 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		int[] flatImageArray = new int[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asInts().read(flatImageArray);
		return ArrayImgs.ints(flatImageArray, tensorShape);
    }

    private static ArrayImg<FloatType, FloatArray> buildFromTensorFloat(TFloat32 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		float[] flatImageArray = new float[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asFloats().read(flatImageArray);
		return ArrayImgs.floats(flatImageArray, tensorShape);
    }

    private static ArrayImg<DoubleType, DoubleArray> buildFromTensorDouble(TFloat64 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		double[] flatImageArray = new double[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asDoubles().read(flatImageArray);
		return ArrayImgs.doubles(flatImageArray, tensorShape);
    }

    private static ArrayImg<LongType, LongArray> buildFromTensorLong(TInt64 tensor)
    {
		long[] tensorShape = tensor.shape().asArray();
		long size = 1;
		for (long ss : tensorShape) {size *= ss;}
		long[] flatImageArray = new long[(int) size];
		// Copy data from tensor to array
        tensor.asRawTensor().data().asLongs().read(flatImageArray);
		return ArrayImgs.longs(flatImageArray, tensorShape);
    }
}

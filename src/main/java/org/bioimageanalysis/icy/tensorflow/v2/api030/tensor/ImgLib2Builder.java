package org.bioimageanalysis.icy.tensorflow.v2.api030.tensor;

import org.bioimageanalysis.icy.deeplearning.utils.IndexingUtils;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TFloat64;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TInt64;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import net.imglib2.Cursor;
import net.imglib2.img.Img;
import net.imglib2.img.ImgFactory;
import net.imglib2.img.cell.CellImgFactory;
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

    private static Img<ByteType> buildFromTensorByte(TUint8 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ImgFactory< ByteType > factory = new CellImgFactory<>( new ByteType(), 5 );
        final Img< ByteType > outputImg = (Img<ByteType>) factory.create(tensorShape);
    	Cursor<ByteType> tensorCursor= outputImg.cursor();
		int totalSize = 1;
		for (long i : tensorShape) {totalSize *= i;}
        byte[] flatArr = new byte[totalSize];
        tensor.asRawTensor().data().read(flatArr);
		while (tensorCursor.hasNext()) {
			tensorCursor.fwd();
			long[] cursorPos = tensorCursor.positionAsLongArray();
        	int flatPos = IndexingUtils.multidimensionalIntoFlatIndex(cursorPos, tensorShape);
        	byte val = flatArr[flatPos];
        	tensorCursor.get().set(val);
		}
	 	return outputImg;
    }

    private static Img<IntType> buildFromTensorInt(TInt32 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ImgFactory< IntType > factory = new CellImgFactory<>( new IntType(), 5 );
        final Img< IntType > outputImg = (Img<IntType>) factory.create(tensorShape);
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

    private static Img<FloatType> buildFromTensorFloat(TFloat32 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ImgFactory< FloatType > factory = new CellImgFactory<>( new FloatType(), 5 );
        final Img< FloatType > outputImg = (Img<FloatType>) factory.create(tensorShape);
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

    private static Img<DoubleType> buildFromTensorDouble(TFloat64 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ImgFactory< DoubleType > factory = new CellImgFactory<>( new DoubleType(), 5 );
        final Img< DoubleType > outputImg = (Img<DoubleType>) factory.create(tensorShape);
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

    private static Img<LongType> buildFromTensorLong(TInt64 tensor)
    {
    	long[] tensorShape = tensor.shape().asArray();
    	final ImgFactory< LongType > factory = new CellImgFactory<>( new LongType(), 5 );
        final Img< LongType > outputImg = (Img<LongType>) factory.create(tensorShape);
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

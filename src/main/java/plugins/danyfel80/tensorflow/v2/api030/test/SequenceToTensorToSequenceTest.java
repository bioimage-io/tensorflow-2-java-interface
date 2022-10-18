package plugins.danyfel80.tensorflow.v2.api030.test;

import org.bioimageanalysis.icy.tensorflow.v2.api030.tensor.Nd4jBuilder;
import org.bioimageanalysis.icy.tensorflow.v2.api030.tensor.TensorBuilder;
import org.tensorflow.ConcreteFunction;
import org.tensorflow.Signature;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.math.Div;
import org.tensorflow.types.TUint8;
import org.tensorflow.types.family.TType;

import icy.main.Icy;
import icy.plugin.PluginLauncher;
import icy.plugin.PluginLoader;
import icy.sequence.Sequence;
import plugins.adufour.ezplug.EzPlug;
import plugins.adufour.ezplug.EzVarSequence;

/**
 * @author Daniel Felipe Gonzalez Obando
 */
public class SequenceToTensorToSequenceTest extends EzPlug
{

    private EzVarSequence varInSequence;

    @Override
    protected void initialize()
    {
        varInSequence = new EzVarSequence("Sequence");
        addEzComponent(varInSequence);
    }

    @Override
    protected void execute()
    {
        Sequence sequence = varInSequence.getValue(true);
        int[] tensorDimOrder = new int[] {0, 1, 2, 3, 4};
        int[] tensorDimOrder1 = new int[] {0, 1, 2, 3, 4};
        long tStart = 0, tTensor = 0, tTensorResult = 0, tResult = 0;
        tStart = System.currentTimeMillis();
        try (TType tensor = TensorBuilder.build(sequence, tensorDimOrder))
        {
            tTensor = System.currentTimeMillis();
            try (ConcreteFunction div = ConcreteFunction.create(SequenceToTensorToSequenceTest::div);
                    TUint8 divResult = (TUint8) div.call(tensor))
            {
                tTensorResult = System.currentTimeMillis();
                Sequence rebuiltSequence = Nd4jBuilder.build(divResult, tensorDimOrder1);
                tResult = System.currentTimeMillis();
                addSequence(rebuiltSequence);
            }
        }
        long tConversion1 = tTensor - tStart;
        long tProcess = tTensorResult - tTensor;
        long tConversion2 = tResult - tTensorResult;
        System.out.println("Conversion to tensor = " + tConversion1 + "msec");
        System.out.println("Processing = " + tProcess + "msec");
        System.out.println("Conversion to sequence = " + tConversion2 + "msec");
    }

    private static Signature div(Ops tf)
    {
        Placeholder<TUint8> x = tf.placeholder(TUint8.class);
        Div<TUint8> dblX = tf.math.div(x, tf.constant((byte) 2));
        return Signature.builder().input("x", x).output("dbl", dblX).build();
    }

    @Override
    public void clean()
    {
    }

    public static void main(String[] args)
    {
        Icy.main(args);
        PluginLauncher.start(PluginLoader.getPlugin(SequenceToTensorToSequenceTest.class.getName()));
    }

}

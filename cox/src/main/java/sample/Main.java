package sample;

import javafx.application.Application;
import javafx.fxml.FXML;
import javafx.scene.Scene;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.control.TextField;
import javafx.scene.layout.Pane;
import javafx.stage.Stage;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.recurrent.LastTimeStep;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.LabelLastTimeStepPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.AdaGrad;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;


import java.io.*;
import java.util.List;

public class Main extends Application {
    @FXML
    private TextField trainMessage;
    @FXML
    private TextField predictMessage;
    @FXML
    private Button trainButton;
    @FXML
    private Button predictButton;
    @FXML
    private TextField t1;
    @FXML
    private TextField t2;
    @FXML
    private TextField t3;
    @FXML
    private TextField t4;
    @FXML
    private TextField t5;
    @FXML
    private TextField t6;
    @FXML
    private TextField t7;
    @FXML
    private TextField t8;
    @FXML
    private TextField t9;
    @FXML
    private TextField t10;

    private DataSet data;
    private MultiLayerNetwork myNet;
    public double[][] getInputs()
    {
        return new double[1][10];


    }
    public void get_data() throws IOException {
        int currentRow=0;
        INDArray input = Nd4j.zeros(200, 10);
        INDArray output = Nd4j.zeros(200, 2);
        BufferedReader in = new BufferedReader(new FileReader("heart.csv"));

        String line = null;
        while (line == in.readLine())
        {
            String[] args=line.split(",");
            input.putScalar(new int[]{currentRow, 0}, Integer.parseInt(args[0]));
            input.putScalar(new int[]{currentRow, 1}, Integer.parseInt(args[1]));
            input.putScalar(new int[]{currentRow, 2}, Integer.parseInt(args[2]));
            input.putScalar(new int[]{currentRow, 3}, Integer.parseInt(args[3]));
            input.putScalar(new int[]{currentRow, 4}, Integer.parseInt(args[4]));
            input.putScalar(new int[]{currentRow, 5}, Integer.parseInt(args[5]));
            input.putScalar(new int[]{currentRow, 6}, Integer.parseInt(args[6]));
            input.putScalar(new int[]{currentRow, 7}, Integer.parseInt(args[7]));
            input.putScalar(new int[]{currentRow, 8}, Integer.parseInt(args[8]));
            input.putScalar(new int[]{currentRow, 9}, Integer.parseInt(args[9]));

            output.putScalar(new int[]{currentRow, 0}, Integer.parseInt(args[10]));
            output.putScalar(new int[]{currentRow, 1}, Integer.parseInt(args[11]));
            currentRow+=1;

        }
        this.data=new DataSet(input,output);
    }
    public void train() throws IOException, InterruptedException {
        this.trainMessage.setText("In the process...");
        this.get_data();


        MultiLayerConfiguration conf = NetBuild.getConfig();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(Constants.iterationListener));
        for( int i=0; i < Constants.epochNumber; i++ ) {
            model.fit(this.data);
        }
        //record score once every 100 iterations

       this.myNet=model;
        this.trainMessage.setText("Done Training!");

    }
    public void predict()
    {
        double[][] inputs=this.getInputs();

        inputs[0][0]= Integer.parseInt(this.t1.getText());
        inputs[0][1]= Integer.parseInt(this.t2.getText());
        inputs[0][2]= Integer.parseInt(this.t3.getText());
        inputs[0][3]= Integer.parseInt(this.t4.getText());
        inputs[0][4]= Integer.parseInt(this.t5.getText());
        inputs[0][5]= Integer.parseInt(this.t6.getText());
        inputs[0][6]= Integer.parseInt(this.t7.getText());
        inputs[0][7]= Integer.parseInt(this.t8.getText());
        inputs[0][8]= Integer.parseInt(this.t9.getText());
        inputs[0][9]= Double.parseDouble(this.t10.getText());

        INDArray prediction = this.myNet.output(new NDArray(inputs));
        System.out.println(prediction.getDouble(1));
        System.out.println(prediction.getDouble(0));
        System.out.println(prediction.getDouble(1)  / prediction.getDouble(0));

        if (prediction.getDouble(1)  / prediction.getDouble(0) > Constants.threshhold)
        {
            this.predictMessage.setText("You are ok!" + prediction.getDouble(1)  / prediction.getDouble(0));
        }
        else
        {
            this.predictMessage.setText("Go see a doctor!!" + prediction.getDouble(1)  / prediction.getDouble(0));
        }
    }
    @Override
    public void start(Stage primaryStage) throws Exception{
        //Parent root = FXMLLoader.load(getClass().getResource("../../../../sample.fxml"));
        primaryStage.setTitle("Heart diesese prediction");
        //<Pane maxHeight="-Infinity" maxWidth="-Infinity" minHeight="-Infinity" minWidth="-Infinity" prefHeight="500.0" prefWidth="599.0" xmlns="http://javafx.com/javafx/15.0.1" xmlns:fx="http://javafx.com/fxml/1" fx:controller= "sample.Controller">
        Pane layout = new Pane();
        layout.setMaxHeight(999999999);
        layout.setMinHeight(999999999);
        layout.setMaxWidth(999999999);
        layout.setMaxWidth(999999999);
        layout.setPrefHeight(500);
        layout.setMaxWidth(600);

        //primaryStage.setScene(new Scene(root, 300, 275));

        Button b=new Button();

        b.setId("trainButton");
        b.setText("Train");
        b.setLayoutX(364.0);
        b.setLayoutY(159.0);
        b.setPrefHeight(25.0);
        b.setPrefWidth(124.0);
        b.setOnMouseClicked(mouseEvent -> {
            try {
                this.train();
            } catch (IOException e) {
                e.printStackTrace();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        this.trainButton=b;

        layout.getChildren().add(b);
        //<Label fx:id="ageLabel" layoutX="66.0" layoutY="123.0" text="Age" />
        Label l1=new Label();
        l1.setId("ageLabel");
        l1.setLayoutX(66.0);
        l1.setLayoutY(123.0);
        l1.setText("Age");
        layout.getChildren().add(l1);




//      <Label fx:id="genderLabel" layoutX="57.0" layoutY="154.0" text="Gender" />

        Label l2=new Label();
        l2.setId("genderLabel");
        l2.setLayoutX(57.0);
        l2.setLayoutY(154.0);
        l2.setText("Gender");
        layout.getChildren().add(l2);



//      <Label layoutX="41.0" layoutY="182.0" text="Pain type" />

        Label l3=new Label();
        l3.setId("painLabel");
        l3.setLayoutX(41.0);
        l3.setLayoutY(182.0);
        l3.setText("PainType");
        layout.getChildren().add(l3);

//      <Label fx:id="bloodPressureLabel" layoutX="19.0" layoutY="209.0" prefHeight="17.0" prefWidth="79.0" text="Blood pressure" />
        Label l4=new Label();
        l4.setId("bloodLabel");
        l4.setLayoutX(19.0);
        l4.setLayoutY(209.0);
        l4.setText("Blood pressure");
        layout.getChildren().add(l4);


//      <Label fx:id="cholesterolLabel" layoutX="26.0" layoutY="242.0" prefHeight="17.0" prefWidth="65.0" text="Cholesterol" />
        Label l5=new Label();
        l5.setId("cholesteronLabel");
        l5.setLayoutX(26.0);
        l5.setLayoutY(242.0);
        l5.setText("Cholesterol");
        layout.getChildren().add(l5);

//      <Label fx:id="bloodSugarLabel" layoutX="26.0" layoutY="276.0" prefHeight="17.0" prefWidth="65.0" text="Blood sugar" />
        Label l6=new Label();
        l6.setId("sugarLabel");
        l6.setLayoutX(26.0);
        l6.setLayoutY(276.0);
        l6.setText("Sugar");
        layout.getChildren().add(l6);

//      <Label fx:id="electroLabel" layoutX="21.0" layoutY="306.0" text="Electro-results" />
        Label l7=new Label();
        l7.setId("painLabel");
        l7.setLayoutX(21.0);
        l7.setLayoutY(306.0);
        l7.setText("Electro");
        layout.getChildren().add(l7);

//      <Label fx:id="maxBMPLabel" layoutX="35.0" layoutY="337.0" text="maxBPM" />
        Label l8=new Label();
        l8.setId("BPMLabel");
        l8.setLayoutX(35.0);
        l8.setLayoutY(337.0);
        l8.setText("MaxBMP");
        layout.getChildren().add(l8);

//      <Label fx:id="anginaLabel" layoutX="41.0" layoutY="369.0" text="angina" />
        Label l9=new Label();
        l9.setId("anginaLabel");
        l9.setLayoutX(41.0);
        l9.setLayoutY(369.0);
        l9.setText("Angina");
        layout.getChildren().add(l9);

//      <Label fx:id="previousPeakLabel" layoutX="7.0" layoutY="408.0" text="previousPeakLabel" />
        Label l10=new Label();
        l10.setId("peakLabel");
        l10.setLayoutX(7.0);
        l10.setLayoutY(408.0);
        l10.setText("PreviousPeak");
        layout.getChildren().add(l10);







//      <TextField fx:id="ageField" layoutX="142.0" layoutY="119.0" prefHeight="25.0" prefWidth="149.0" />
        TextField t1=new TextField();
        t1.setText("");
        t1.setLayoutX(142);
        t1.setLayoutY(119);
        layout.getChildren().add(t1);
        this.t1=t1;
//      <TextField fx:id="genderField" layoutX="142.0" layoutY="150.0" />
        TextField t2=new TextField();
        t2.setText("");
        t2.setLayoutX(142);
        t2.setLayoutY(150);
        layout.getChildren().add(t2);
        this.t2=t2;
//      <TextField fx:id="painField" layoutX="142.0" layoutY="178.0" />
        TextField t3=new TextField();
        t3.setText("");
        t3.setLayoutX(142);
        t3.setLayoutY(178);
        layout.getChildren().add(t3);
        this.t3=t3;

//      <TextField fx:id="bloodPressureField" layoutX="142.0" layoutY="205.0" />
        TextField t4=new TextField();
        t4.setText("");
        t4.setLayoutX(142);
        t4.setLayoutY(205);
        layout.getChildren().add(t4);
        this.t4=t4;
//      <TextField fx:id="cholesterolField" layoutX="142.0" layoutY="238.0" />
        TextField t5=new TextField();
        t5.setText("");
        t5.setLayoutX(142);
        t5.setLayoutY(238);
        layout.getChildren().add(t5);
        this.t5=t5;
//      <TextField fx:id="bloodSugarField" layoutX="142.0" layoutY="272.0" />
        TextField t6=new TextField();
        t6.setText("");
        t6.setLayoutX(142);
        t6.setLayoutY(272);
        layout.getChildren().add(t6);
        this.t6=t6;
//      <TextField fx:id="electroField" layoutX="142.0" layoutY="302.0" />
        TextField t7=new TextField();
        t7.setText("");
        t7.setLayoutX(142);
        t7.setLayoutY(302);
        layout.getChildren().add(t7);
        this.t7=t7;
//      <TextField fx:id="BPMField" layoutX="142.0" layoutY="333.0" />
        TextField t8=new TextField();
        t8.setText("");
        t8.setLayoutX(142);
        t8.setLayoutY(333);
        layout.getChildren().add(t8);
        this.t8=t8;
//      <TextField fx:id="anginaField" layoutX="142.0" layoutY="365.0" />
        TextField t9=new TextField();
        t9.setText("");
        t9.setLayoutX(142);
        t9.setLayoutY(365);
        this.t9=t9;
        layout.getChildren().add(t9);
//      <TextField fx:id="peakField" layoutX="142.0" layoutY="404.0" />
        TextField t10=new TextField();
        t10.setText("");
        t10.setLayoutX(142);
        t10.setLayoutY(404);
        layout.getChildren().add(t10);
        this.t10=t10;
//      <TextField fx:id="trainMessageField" layoutX="364.0" layoutY="187.0" promptText="Not yet trained!" />
        TextField t11=new TextField();
        t11.setText("Not yet trained!");
        t11.setLayoutX(364);
        t11.setLayoutY(187);
        layout.getChildren().add(t11);
        this.trainMessage=t11;
//      <Button fx:id="predictButton" layoutX="364.0" layoutY="285.0" mnemonicParsing="false" prefHeight="25.0" prefWidth="149.0" text="Predict" />
        Button predict=new Button();
        predict.setLayoutX(364);
        predict.setLayoutY(285);
        predict.setText("Predict");
        predict.setOnMouseClicked(mouseEvent -> {this.predict();});
        layout.getChildren().add(predict);
//      <TextField fx:id="predictionField" layoutX="364.0" layoutY="315.0" promptText="Nothing to predict!" />
        TextField t12=new TextField();
        t12.setText("No prediction yet!!");
        t12.setLayoutX(364);
        t12.setLayoutY(315);
        layout.getChildren().add(t12);
        this.predictMessage=t12;

        Scene scene=new Scene(layout,700,700);
        primaryStage.setScene(scene);
        primaryStage.show();
    }


    public static void main(String[] args) {
        launch(args);
    }
}

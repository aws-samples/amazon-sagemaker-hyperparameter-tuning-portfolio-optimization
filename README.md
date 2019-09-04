## amazon-sagemaker-hyperparameter-tuning-portfolio-optimization

Financial institutions that extend credit face the dual tasks of evaluating the credit risk associated with each loan application and of determining a threshold that defines the level of risk they are willing to take on. The evaluation of credit risk is a familiar application of Machine Learning classification models. The determination of a classification threshold, though, is oftentimes treated as a secondary concern and set in an ad-hoc, unprincipled manner. As a result, institutions may be creating underperforming portfolios and leaving risk-adjusted return on the table.

In this blog post, we describe how to use Amazon SageMaker Automatic Model Tuning functionality to determine the classification threshold that maximizes the portfolio value of a lender choosing which subset of borrowers to extend credit to. More generally, we describe a method of choosing an optimal threshold, or set of thresholds, in a classification setting. The method we describe doesn't rely on rules of thumb or generic metrics. It is a systematic and principled method that relies on a business success metric specific to the problem at hand.

The method is based upon utility theory and the idea that a rational individual makes decision in such way as to maximize her expected utility, or subjective value. In this post, we assume that the lender is attempting to maximize the expected dollar value of her portfolio, that is, her expected utility, by choosing a classification threshold that divides loan applications into two groups: those she accepts and lends to, and those she rejects. In other words, the lender is searching over the space of potential threshold values in order to find the threshold that results in the highest value for the function that describes her portfolio value. 

This blog uses Amazon SageMaker Automatic Model Tuning to find that optimal threshold. This is a novel use of the Automatic Model Tuning function, which is typically used to choose the hyperparameters that optimize model performance. Here instead we use it as a general tool to maximize a function over some specific parameter space.

This approach has several advantages over the typical threshold determination approach. Typically, a classification threshold is set (or allowed to default) to 0.5, and this threshold doesn't in fact generate the maximum possible result in the majority of use cases. The approach described here chooses a threshold that generates the maximum possible result for the specific business use case being addressed. 

Also, this approach moves beyond using general rules-of-thumb and "expert judgment" in determining an optimal threshold. It lays out a structured framework that can be systematically applied to any classification problem. Additionally, this approach requires the business to explicitly state its cost matrix, based on the specific actions to be taken on the predictions and their benefits and costs. This evaluation process moves well beyond simply assessing the classification results of the model. This approach can drive challenging discussions in the business, and force differing implicit decisions and valuations onto the table for open discussion and agreement. This drives the discussion from a simple “maximize this value”, to a more informative analysis that allows more complex economic tradeoffs to be made – providing more value back to the business.

To build and train the model, open the jupyter notebook contained in the hyperparameter-tuning-portfolio-optimization folder and follow the instructions in the notebook.

The blog post associated with this repo is located here.

## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.

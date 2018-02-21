


Terminology
-----------

+-------------+----------------------------------------------+------+--------------+
|Name         |Definition                                    | Type | Class        |
+=============+==============================================+======+==============+
|spacer       |The (usually) 20 bp RNA segment of the gRNA   | RNA  | Bio.Seq      |
|             |which provides homology directed targeting.   |      |              |
+-------------+----------------------------------------------+------+--------------+
| gRNA        | spacer + secondary structure which complexes | RNA  | Bio.Seq      |
|             | with Cas9 to direct targeting.               |      |              |
------------------------------------------------------------------------------------
|target       | The region of the DNA that is bound by Cas9  | DNA  | Bio.Seq      |
|             | as a result of gRNA homology. This segment   |      |              |
|             | is the DNA strand that is homologous to the  |      |              |
|             | `spacer` and includes the PAM sequence.      |      |              |
+-------------+----------------------------------------------+------+--------------+
|locus        | The DNA segment against which `spacers` are  | DNA  | Bio.SeqRecord|
|             | designed. The locus includes the target and  |      |              |
|             | surrounding DNA and typically represents a   |      |              |
|             | region of the genome (ie. a gene) to be      |      |              |
|             | bound by Cas9                                |      |              |
+-------------+----------------------------------------------+------+--------------+
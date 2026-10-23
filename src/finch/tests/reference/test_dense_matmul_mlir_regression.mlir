Compiling MLIR code:
module {
  func.func @main(%_A_15: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.ptr, %_A_7: !llvm.ptr, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_3 = llvm.load %v_2 : !llvm.ptr -> !llvm.ptr
    %v_4 = llvm.getelementptr %v_3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_5 = llvm.load %v_4 : !llvm.ptr -> !llvm.ptr
    %v_6 = llvm.getelementptr %v_5[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_7 = llvm.load %v_6 : !llvm.ptr -> !llvm.ptr
    %v_8 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_9 = llvm.load %v_8 : !llvm.ptr -> !llvm.ptr
    %v_10 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_11 = llvm.load %v_10 : !llvm.ptr -> i64
    %v_12 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = arith.constant 0 : i64
    %v_14 = arith.constant 1 : i64
    %v_15 = llvm.insertvalue %v_9, %v_12[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_16 = llvm.insertvalue %v_9, %v_15[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_17 = llvm.insertvalue %v_13, %v_16[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_18 = llvm.insertvalue %v_11, %v_17[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_19 = llvm.insertvalue %v_14, %v_18[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_20 = builtin.unrealized_conversion_cast %v_19 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_21 = builtin.unrealized_conversion_cast %v_20 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_22 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_21, %v_22 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_23 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_24 = llvm.load %v_23 : !llvm.ptr -> !llvm.ptr
    %v_25 = llvm.getelementptr %v_24[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_26 = llvm.load %v_25 : !llvm.ptr -> i64
    %v_27 = arith.index_cast %v_26 : i64 to index
    %v_28 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_29 = llvm.load %v_28 : !llvm.ptr -> i64
    %v_30 = arith.index_cast %v_29 : i64 to index
    %v_31 = llvm.extractvalue %_A_16[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_32 = llvm.getelementptr %v_31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.getelementptr %v_33[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_35 = llvm.load %v_34 : !llvm.ptr -> !llvm.ptr
    %v_36 = llvm.getelementptr %v_35[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_37 = llvm.load %v_36 : !llvm.ptr -> !llvm.ptr
    %v_38 = llvm.getelementptr %v_37[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_39 = llvm.load %v_38 : !llvm.ptr -> !llvm.ptr
    %v_40 = llvm.getelementptr %v_37[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_41 = llvm.load %v_40 : !llvm.ptr -> i64
    %v_42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_43 = llvm.insertvalue %v_39, %v_42[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.insertvalue %v_39, %v_43[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_45 = llvm.insertvalue %v_13, %v_44[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_46 = llvm.insertvalue %v_41, %v_45[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_47 = llvm.insertvalue %v_14, %v_46[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_48 = builtin.unrealized_conversion_cast %v_47 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_49 = builtin.unrealized_conversion_cast %v_48 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_50 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_49, %v_50 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_51 = llvm.getelementptr %v_31[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_52 = llvm.load %v_51 : !llvm.ptr -> !llvm.ptr
    %v_53 = llvm.getelementptr %v_52[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_54 = llvm.load %v_53 : !llvm.ptr -> i64
    %v_55 = arith.index_cast %v_54 : i64 to index
    %v_56 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_57 = llvm.load %v_56 : !llvm.ptr -> i64
    %v_58 = arith.index_cast %v_57 : i64 to index
    %v_59 = llvm.getelementptr %_A_18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_60 = llvm.load %v_59 : !llvm.ptr -> !llvm.ptr
    %v_61 = llvm.getelementptr %v_60[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_62 = llvm.load %v_61 : !llvm.ptr -> !llvm.ptr
    %v_63 = llvm.getelementptr %v_60[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_64 = llvm.load %v_63 : !llvm.ptr -> i64
    %v_65 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_66 = llvm.insertvalue %v_62, %v_65[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_67 = llvm.insertvalue %v_62, %v_66[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_68 = llvm.insertvalue %v_13, %v_67[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_69 = llvm.insertvalue %v_64, %v_68[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_70 = llvm.insertvalue %v_14, %v_69[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_71 = builtin.unrealized_conversion_cast %v_70 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_72 = builtin.unrealized_conversion_cast %v_71 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_73 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_72, %v_73 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_74 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_75 = llvm.load %v_74 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_76 = llvm.extractvalue %v_75[1] : !llvm.struct<(i64, i64)>
    %v_77 = arith.index_cast %v_76 : i64 to index
    %v_78 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_79 = llvm.load %v_78 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_80 = llvm.extractvalue %v_79[0] : !llvm.struct<(i64, i64)>
    %v_81 = arith.index_cast %v_80 : i64 to index
    %v_82 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_83 = llvm.load %v_82 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_84 = llvm.extractvalue %v_83[1] : !llvm.struct<(i64, i64)>
    %v_85 = arith.index_cast %v_84 : i64 to index
    %v_86 = llvm.getelementptr %_A_7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_87 = llvm.load %v_86 : !llvm.ptr -> !llvm.ptr
    %v_88 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> !llvm.ptr
    %v_90 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_91 = llvm.load %v_90 : !llvm.ptr -> i64
    %v_92 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_93 = llvm.insertvalue %v_89, %v_92[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_94 = llvm.insertvalue %v_89, %v_93[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_95 = llvm.insertvalue %v_13, %v_94[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_96 = llvm.insertvalue %v_91, %v_95[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_97 = llvm.insertvalue %v_14, %v_96[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_98 = builtin.unrealized_conversion_cast %v_97 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_99 = builtin.unrealized_conversion_cast %v_98 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_100 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_99, %v_100 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_101 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_102 = llvm.load %v_101 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_103 = llvm.extractvalue %v_102[0] : !llvm.struct<(i64, i64)>
    %v_104 = arith.index_cast %v_103 : i64 to index
    %v_105 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_106 = llvm.load %v_105 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_107 = llvm.extractvalue %v_106[1] : !llvm.struct<(i64, i64)>
    %v_108 = arith.index_cast %v_107 : i64 to index
    %v_109 = arith.constant 0 : index
    %v_110 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_111 = builtin.unrealized_conversion_cast %v_110 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_112 = memref.dim %v_111, %v_109 : memref<?xf64>
    %v_113 = arith.constant 1 : index
    scf.for %v_114 = %v_109 to %v_112 step %v_113 {
      %v_115 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_116 = builtin.unrealized_conversion_cast %v_115 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_117 = arith.constant 0.0 : f64
      memref.store %v_117, %v_116[%v_114] : memref<?xf64>
    }
    scf.for %v_118 = %v_109 to %v_27 step %v_113 {
      %v_119 = arith.muli %v_81, %v_118 : index
      scf.for %v_120 = %v_109 to %v_30 step %v_113 {
        %v_121 = arith.muli %v_85, %v_120 : index
        %v_122 = arith.addi %v_119, %v_121 : index
        scf.for %v_123 = %v_109 to %v_27 step %v_113 {
          %v_124 = arith.muli %v_120, %v_27 : index
          %v_125 = arith.addi %v_124, %v_123 : index
          %v_126 = arith.cmpi eq, %v_123, %v_118 : index
          scf.if %v_126 {
            %v_127 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_128 = builtin.unrealized_conversion_cast %v_127 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_129 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_130 = builtin.unrealized_conversion_cast %v_129 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_131 = memref.load %v_130[%v_125] : memref<?xf64>
            memref.store %v_131, %v_128[%v_122] : memref<?xf64>
          }
        }
      }
    }
    %v_132 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_133 = builtin.unrealized_conversion_cast %v_132 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_134 = memref.dim %v_133, %v_109 : memref<?xf64>
    scf.for %v_135 = %v_109 to %v_134 step %v_113 {
      %v_136 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_137 = builtin.unrealized_conversion_cast %v_136 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_138 = arith.constant 0.0 : f64
      memref.store %v_138, %v_137[%v_135] : memref<?xf64>
    }
    scf.for %v_139 = %v_109 to %v_58 step %v_113 {
      %v_140 = arith.muli %v_81, %v_139 : index
      scf.for %v_141 = %v_109 to %v_77 step %v_113 {
        %v_142 = arith.muli %v_104, %v_141 : index
        %v_143 = arith.muli %v_85, %v_141 : index
        %v_144 = arith.addi %v_140, %v_143 : index
        scf.for %v_145 = %v_109 to %v_55 step %v_113 {
          %v_146 = arith.muli %v_108, %v_145 : index
          %v_147 = arith.addi %v_142, %v_146 : index
          %v_148 = arith.muli %v_139, %v_55 : index
          %v_149 = arith.addi %v_148, %v_145 : index
          %v_150 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_151 = builtin.unrealized_conversion_cast %v_150 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_152 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_153 = builtin.unrealized_conversion_cast %v_152 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_154 = memref.load %v_153[%v_147] : memref<?xf64>
          %v_155 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_156 = builtin.unrealized_conversion_cast %v_155 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_157 = memref.load %v_156[%v_144] : memref<?xf64>
          %v_158 = llvm.load %v_50 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_159 = builtin.unrealized_conversion_cast %v_158 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_160 = memref.load %v_159[%v_149] : memref<?xf64>
          %v_161 = arith.mulf %v_157, %v_160 : f64
          %v_162 = arith.addf %v_154, %v_161 : f64
          memref.store %v_162, %v_151[%v_147] : memref<?xf64>
        }
      }
    }
    %v_163 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_164 = builtin.unrealized_conversion_cast %v_163 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_165 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_166 = builtin.unrealized_conversion_cast %v_165 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_167 = memref.dim %v_166, %v_109 : memref<?xf64>
    %v_168 = arith.index_cast %v_167 : index to i64
    %v_169 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_168, %v_169 : i64, !llvm.ptr
    %v_170 = memref.extract_aligned_pointer_as_index %v_164 : memref<?xf64> -> index
    %v_171 = arith.index_cast %v_170 : index to i64
    %v_172 = llvm.inttoptr %v_171 : i64 to !llvm.ptr
    %v_173 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_172, %v_173 : !llvm.ptr, !llvm.ptr
    %v_174 = llvm.load %v_50 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_175 = builtin.unrealized_conversion_cast %v_174 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_176 = llvm.load %v_50 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_177 = builtin.unrealized_conversion_cast %v_176 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_178 = memref.dim %v_177, %v_109 : memref<?xf64>
    %v_179 = arith.index_cast %v_178 : index to i64
    %v_180 = llvm.getelementptr %v_37[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_179, %v_180 : i64, !llvm.ptr
    %v_181 = memref.extract_aligned_pointer_as_index %v_175 : memref<?xf64> -> index
    %v_182 = arith.index_cast %v_181 : index to i64
    %v_183 = llvm.inttoptr %v_182 : i64 to !llvm.ptr
    %v_184 = llvm.getelementptr %v_37[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_183, %v_184 : !llvm.ptr, !llvm.ptr
    %v_185 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_186 = builtin.unrealized_conversion_cast %v_185 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_187 = llvm.load %v_73 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_188 = builtin.unrealized_conversion_cast %v_187 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_189 = memref.dim %v_188, %v_109 : memref<?xf64>
    %v_190 = arith.index_cast %v_189 : index to i64
    %v_191 = llvm.getelementptr %v_60[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_190, %v_191 : i64, !llvm.ptr
    %v_192 = memref.extract_aligned_pointer_as_index %v_186 : memref<?xf64> -> index
    %v_193 = arith.index_cast %v_192 : index to i64
    %v_194 = llvm.inttoptr %v_193 : i64 to !llvm.ptr
    %v_195 = llvm.getelementptr %v_60[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_194, %v_195 : !llvm.ptr, !llvm.ptr
    %v_196 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_197 = builtin.unrealized_conversion_cast %v_196 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_198 = llvm.load %v_100 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_199 = builtin.unrealized_conversion_cast %v_198 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_200 = memref.dim %v_199, %v_109 : memref<?xf64>
    %v_201 = arith.index_cast %v_200 : index to i64
    %v_202 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_201, %v_202 : i64, !llvm.ptr
    %v_203 = memref.extract_aligned_pointer_as_index %v_197 : memref<?xf64> -> index
    %v_204 = arith.index_cast %v_203 : index to i64
    %v_205 = llvm.inttoptr %v_204 : i64 to !llvm.ptr
    %v_206 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_205, %v_206 : !llvm.ptr, !llvm.ptr
    %v_207 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_208 = llvm.insertvalue %_A_7, %v_207[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_208, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}

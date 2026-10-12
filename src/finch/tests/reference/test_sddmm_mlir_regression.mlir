Compiling MLIR code:
module {
  func.func @scansearch(
    %arr: memref<?xindex>, %x: index, %lo: index, %hi: index
  ) -> index attributes {llvm.emit_c_interface} {
    %1 = arith.constant 1 : index
    %g:2 = scf.while (%d = %1, %p = %lo) : (index, index) -> (index, index) {
      %plt = arith.cmpi slt, %p, %hi : index
      %cond = scf.if %plt -> (i1) {
        %ap = memref.load %arr[%p] : memref<?xindex>
        %al = arith.cmpi slt, %ap, %x : index
        scf.yield %al : i1
      } else {
        %f = arith.constant false
        scf.yield %f : i1
      }
      scf.condition(%cond) %d, %p : index, index
    } do {
    ^bb0(%d: index, %p: index):
      %d2 = arith.shli %d, %1 : index
      %p2 = arith.addi %p, %d2 : index
      scf.yield %d2, %p2 : index, index
    }
    %lo1 = arith.subi %g#1, %g#0 : index
    %minp = arith.minsi %g#1, %hi : index
    %hi1 = arith.addi %minp, %1 : index
    %b:2 = scf.while (%l = %lo1, %h = %hi1) : (index, index) -> (index, index) {
      %hm1 = arith.subi %h, %1 : index
      %go = arith.cmpi slt, %l, %hm1 : index
      scf.condition(%go) %l, %h : index, index
    } do {
    ^bb0(%l: index, %h: index):
      %diff = arith.subi %h, %l : index
      %half = arith.shrsi %diff, %1 : index
      %m = arith.addi %l, %half : index
      %am = memref.load %arr[%m] : memref<?xindex>
      %al = arith.cmpi slt, %am, %x : index
      %l2, %h2 = scf.if %al -> (index, index) {
        scf.yield %m, %h : index, index
      } else {
        scf.yield %l, %m : index, index
      }
      scf.yield %l2, %h2 : index, index
    }
    return %b#1 : index
  }

  func.func @main(%_A_19: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_20: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_21: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_9: !llvm.ptr, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_3 = llvm.load %v_2 : !llvm.ptr -> !llvm.ptr
    %v_4 = llvm.getelementptr %v_3[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
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
    %v_25 = llvm.getelementptr %v_24[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_26 = llvm.load %v_25 : !llvm.ptr -> i64
    %v_27 = arith.index_cast %v_26 : i64 to index
    %v_28 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_29 = llvm.load %v_28 : !llvm.ptr -> !llvm.ptr
    %v_30 = llvm.getelementptr %v_29[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_31 = llvm.load %v_30 : !llvm.ptr -> !llvm.ptr
    %v_32 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_35 = llvm.load %v_34 : !llvm.ptr -> i64
    %v_36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_33, %v_36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_33, %v_37[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = llvm.insertvalue %v_13, %v_38[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_40 = llvm.insertvalue %v_35, %v_39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = llvm.insertvalue %v_14, %v_40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_42 = builtin.unrealized_conversion_cast %v_41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_43 = builtin.unrealized_conversion_cast %v_42 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_43, %v_44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_45 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_46 = llvm.load %v_45 : !llvm.ptr -> !llvm.ptr
    %v_47 = llvm.getelementptr %v_46[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_48 = llvm.load %v_47 : !llvm.ptr -> !llvm.ptr
    %v_49 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_50 = llvm.load %v_49 : !llvm.ptr -> !llvm.ptr
    %v_51 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_52 = llvm.load %v_51 : !llvm.ptr -> i64
    %v_53 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_54 = llvm.insertvalue %v_50, %v_53[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_55 = llvm.insertvalue %v_50, %v_54[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_56 = llvm.insertvalue %v_13, %v_55[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_57 = llvm.insertvalue %v_52, %v_56[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_58 = llvm.insertvalue %v_14, %v_57[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_59 = builtin.unrealized_conversion_cast %v_58 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_60 = builtin.unrealized_conversion_cast %v_59 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_61 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_60, %v_61 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_62 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_63 = llvm.load %v_62 : !llvm.ptr -> !llvm.ptr
    %v_64 = llvm.getelementptr %v_63[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_65 = llvm.load %v_64 : !llvm.ptr -> i64
    %v_66 = arith.index_cast %v_65 : i64 to index
    %v_67 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_68 = llvm.load %v_67 : !llvm.ptr -> !llvm.ptr
    %v_69 = llvm.getelementptr %v_68[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_70 = llvm.load %v_69 : !llvm.ptr -> i64
    %v_71 = arith.index_cast %v_70 : i64 to index
    %v_72 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_73 = llvm.load %v_72 : !llvm.ptr -> !llvm.ptr
    %v_74 = llvm.getelementptr %v_73[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_75 = llvm.load %v_74 : !llvm.ptr -> i64
    %v_76 = arith.index_cast %v_75 : i64 to index
    %v_77 = llvm.getelementptr %v[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_78 = llvm.load %v_77 : !llvm.ptr -> !llvm.ptr
    %v_79 = llvm.getelementptr %v_78[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_80 = llvm.load %v_79 : !llvm.ptr -> i1
    %v_81 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_82 = llvm.load %v_81 : !llvm.ptr -> i64
    %v_83 = arith.index_cast %v_82 : i64 to index
    %v_84 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_85 = llvm.load %v_84 : !llvm.ptr -> i64
    %v_86 = arith.index_cast %v_85 : i64 to index
    %v_87 = llvm.extractvalue %_A_20[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_88 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> !llvm.ptr
    %v_90 = llvm.getelementptr %v_89[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_91 = llvm.load %v_90 : !llvm.ptr -> !llvm.ptr
    %v_92 = llvm.getelementptr %v_91[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_93 = llvm.load %v_92 : !llvm.ptr -> !llvm.ptr
    %v_94 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_95 = llvm.load %v_94 : !llvm.ptr -> !llvm.ptr
    %v_96 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_97 = llvm.load %v_96 : !llvm.ptr -> i64
    %v_98 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_99 = llvm.insertvalue %v_95, %v_98[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_100 = llvm.insertvalue %v_95, %v_99[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_101 = llvm.insertvalue %v_13, %v_100[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_102 = llvm.insertvalue %v_97, %v_101[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_103 = llvm.insertvalue %v_14, %v_102[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_104 = builtin.unrealized_conversion_cast %v_103 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_105 = builtin.unrealized_conversion_cast %v_104 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_106 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_105, %v_106 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_107 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_108 = llvm.load %v_107 : !llvm.ptr -> !llvm.ptr
    %v_109 = llvm.getelementptr %v_108[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_110 = llvm.load %v_109 : !llvm.ptr -> i64
    %v_111 = arith.index_cast %v_110 : i64 to index
    %v_112 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_113 = llvm.load %v_112 : !llvm.ptr -> !llvm.ptr
    %v_114 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_115 = llvm.load %v_114 : !llvm.ptr -> i64
    %v_116 = arith.index_cast %v_115 : i64 to index
    %v_117 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_118 = llvm.load %v_117 : !llvm.ptr -> i64
    %v_119 = arith.index_cast %v_118 : i64 to index
    %v_120 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_121 = llvm.load %v_120 : !llvm.ptr -> i64
    %v_122 = arith.index_cast %v_121 : i64 to index
    %v_123 = llvm.extractvalue %_A_21[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_124 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_125 = llvm.load %v_124 : !llvm.ptr -> !llvm.ptr
    %v_126 = llvm.getelementptr %v_125[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_127 = llvm.load %v_126 : !llvm.ptr -> !llvm.ptr
    %v_128 = llvm.getelementptr %v_127[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr)>
    %v_129 = llvm.load %v_128 : !llvm.ptr -> !llvm.ptr
    %v_130 = llvm.getelementptr %v_129[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_131 = llvm.load %v_130 : !llvm.ptr -> !llvm.ptr
    %v_132 = llvm.getelementptr %v_129[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_133 = llvm.load %v_132 : !llvm.ptr -> i64
    %v_134 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_135 = llvm.insertvalue %v_131, %v_134[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_136 = llvm.insertvalue %v_131, %v_135[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_137 = llvm.insertvalue %v_13, %v_136[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_138 = llvm.insertvalue %v_133, %v_137[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_139 = llvm.insertvalue %v_14, %v_138[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_140 = builtin.unrealized_conversion_cast %v_139 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_141 = builtin.unrealized_conversion_cast %v_140 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_142 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_141, %v_142 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_143 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_144 = llvm.load %v_143 : !llvm.ptr -> !llvm.ptr
    %v_145 = llvm.getelementptr %v_144[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_146 = llvm.load %v_145 : !llvm.ptr -> i64
    %v_147 = arith.index_cast %v_146 : i64 to index
    %v_148 = llvm.getelementptr %v_123[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_149 = llvm.load %v_148 : !llvm.ptr -> !llvm.ptr
    %v_150 = llvm.getelementptr %v_149[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_151 = llvm.load %v_150 : !llvm.ptr -> i64
    %v_152 = arith.index_cast %v_151 : i64 to index
    %v_153 = llvm.getelementptr %v_123[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_154 = llvm.load %v_153 : !llvm.ptr -> i64
    %v_155 = arith.index_cast %v_154 : i64 to index
    %v_156 = llvm.getelementptr %v_123[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_157 = llvm.load %v_156 : !llvm.ptr -> i64
    %v_158 = arith.index_cast %v_157 : i64 to index
    %v_159 = llvm.getelementptr %_A_9[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_160 = llvm.load %v_159 : !llvm.ptr -> !llvm.ptr
    %v_161 = llvm.getelementptr %v_160[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_162 = llvm.load %v_161 : !llvm.ptr -> !llvm.ptr
    %v_163 = llvm.getelementptr %v_160[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_164 = llvm.load %v_163 : !llvm.ptr -> i64
    %v_165 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_166 = llvm.insertvalue %v_162, %v_165[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_167 = llvm.insertvalue %v_162, %v_166[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_168 = llvm.insertvalue %v_13, %v_167[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_169 = llvm.insertvalue %v_164, %v_168[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_170 = llvm.insertvalue %v_14, %v_169[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_171 = builtin.unrealized_conversion_cast %v_170 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_172 = builtin.unrealized_conversion_cast %v_171 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_173 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_172, %v_173 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_174 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_175 = llvm.load %v_174 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_176 = llvm.extractvalue %v_175[0] : !llvm.struct<(i64, i64)>
    %v_177 = arith.index_cast %v_176 : i64 to index
    %v_178 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_179 = llvm.load %v_178 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_180 = llvm.extractvalue %v_179[1] : !llvm.struct<(i64, i64)>
    %v_181 = arith.index_cast %v_180 : i64 to index
    %v_182 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_183 = llvm.load %v_182 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_184 = llvm.extractvalue %v_183[0] : !llvm.struct<(i64, i64)>
    %v_185 = arith.index_cast %v_184 : i64 to index
    %v_186 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_187 = llvm.load %v_186 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_188 = llvm.extractvalue %v_187[1] : !llvm.struct<(i64, i64)>
    %v_189 = arith.index_cast %v_188 : i64 to index
    %v_190 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_191 = arith.index_cast %v_83 : index to i64
    %v_192 = llvm.insertvalue %v_191, %v_190[0] : !llvm.struct<(i64, i64)>
    %v_193 = arith.index_cast %v_27 : index to i64
    %v_194 = llvm.insertvalue %v_193, %v_192[1] : !llvm.struct<(i64, i64)>
    %v_195 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_194, %v_195 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_196 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_197 = llvm.load %v_196 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_198 = llvm.extractvalue %v_197[0] : !llvm.struct<(i64, i64)>
    %v_199 = arith.index_cast %v_198 : i64 to index
    %v_200 = llvm.getelementptr %_A_9[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_201 = llvm.load %v_200 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_202 = llvm.extractvalue %v_201[1] : !llvm.struct<(i64, i64)>
    %v_203 = arith.index_cast %v_202 : i64 to index
    %v_204 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_205 = arith.index_cast %v_27 : index to i64
    %v_206 = llvm.insertvalue %v_205, %v_204[0] : !llvm.struct<(i64, i64)>
    %v_207 = arith.constant 1 : index
    %v_208 = arith.index_cast %v_207 : index to i64
    %v_209 = llvm.insertvalue %v_208, %v_206[1] : !llvm.struct<(i64, i64)>
    %v_210 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_209, %v_210 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_211 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_212 = llvm.load %v_211 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_213 = llvm.extractvalue %v_212[0] : !llvm.struct<(i64, i64)>
    %v_214 = arith.index_cast %v_213 : i64 to index
    %v_215 = llvm.getelementptr %_A_9[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_216 = llvm.load %v_215 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_217 = llvm.extractvalue %v_216[1] : !llvm.struct<(i64, i64)>
    %v_218 = arith.index_cast %v_217 : i64 to index
    %v_219 = arith.muli %v_83, %v_27 : index
    %v_220 = arith.index_cast %v_219 : index to i64
    %v_221 = llvm.getelementptr %v_160[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_222 = llvm.getelementptr %v_160[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_223 = llvm.load %v_222 : !llvm.ptr -> !llvm.ptr
    %v_224 = llvm.call %v_223(%v_221, %v_220) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_225 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_226 = llvm.insertvalue %v_224, %v_225[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_227 = llvm.insertvalue %v_224, %v_226[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_228 = llvm.insertvalue %v_13, %v_227[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_229 = llvm.insertvalue %v_220, %v_228[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_230 = llvm.insertvalue %v_14, %v_229[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_231 = builtin.unrealized_conversion_cast %v_230 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_232 = builtin.unrealized_conversion_cast %v_231 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %v_232, %v_173 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_233 = arith.constant 0 : index
    %v_234 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_235 = builtin.unrealized_conversion_cast %v_234 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_236 = memref.dim %v_235, %v_233 : memref<?xf64>
    scf.for %v_237 = %v_233 to %v_236 step %v_207 {
      %v_238 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_239 = builtin.unrealized_conversion_cast %v_238 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_240 = arith.constant 0.0 : f64
      memref.store %v_240, %v_239[%v_237] : memref<?xf64>
    }
    scf.for %v_241 = %v_233 to %v_83 step %v_207 {
      %v_242 = arith.muli %v_214, %v_241 : index
      scf.for %v_243 = %v_233 to %v_111 step %v_207 {
        %v_244 = arith.muli %v_241, %v_111 : index
        %v_245 = arith.addi %v_244, %v_243 : index
        %v_246 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_247 = builtin.unrealized_conversion_cast %v_246 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_248 = memref.load %v_247[%v_241] : memref<?xindex>
        %v_249 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_250 = builtin.unrealized_conversion_cast %v_249 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_251 = arith.addi %v_207, %v_241 : index
        %v_252 = memref.load %v_250[%v_251] : memref<?xindex>
        %v_253 = arith.cmpi slt, %v_248, %v_252 : index
        %v_261, %v_262 = scf.if %v_253 -> (index, index) {
          %v_254 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_255 = builtin.unrealized_conversion_cast %v_254 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_256 = memref.load %v_255[%v_248] : memref<?xindex>
          %v_257 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_258 = builtin.unrealized_conversion_cast %v_257 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_259 = arith.subi %v_252, %v_207 : index
          %v_260 = memref.load %v_258[%v_259] : memref<?xindex>
          scf.yield %v_256, %v_260 : index, index
        } else {
          scf.yield %v_207, %v_233 : index, index
        }
        %v_263 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_264 = builtin.unrealized_conversion_cast %v_263 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_265 = memref.load %v_264[%v_248] : memref<?xindex>
        %v_266 = arith.cmpi slt, %v_265, %v_233 : index
        %v_271 = scf.if %v_266 -> (index) {
          %v_267 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_268 = builtin.unrealized_conversion_cast %v_267 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_269 = arith.subi %v_252, %v_207 : index
          %v_270 = func.call @scansearch(%v_268, %v_233, %v_248, %v_269) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_270 : index
        } else {
          scf.yield %v_248 : index
        }
        %v_319:3 = scf.while (%v_272 = %v_233, %v_273 = %v_271, %v_274 = %v_261) : (index, index, index) -> (index, index, index) {
          %v_275 = arith.addi %v_207, %v_262 : index
          %v_276 = arith.minsi %v_27, %v_275 : index
          %v_277 = arith.cmpi slt, %v_274, %v_276 : index
          scf.condition(%v_277) %v_272, %v_273, %v_274 : index, index, index
        } do {
          ^bb(%v_272: index, %v_273: index, %v_274: index):
          %v_278 = arith.addi %v_207, %v_274 : index
          %v_279 = arith.minsi %v_278, %v_274 : index
          scf.for %v_280 = %v_272 to %v_279 step %v_207 {
            %v_281 = arith.muli %v_218, %v_280 : index
            %v_282 = arith.addi %v_242, %v_281 : index
            %v_283 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_284 = builtin.unrealized_conversion_cast %v_283 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_285 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_286 = builtin.unrealized_conversion_cast %v_285 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_287 = memref.load %v_286[%v_282] : memref<?xf64>
            memref.store %v_287, %v_284[%v_282] : memref<?xf64>
          }
          %v_288 = arith.maxsi %v_272, %v_274 : index
          %v_289 = arith.addi %v_207, %v_274 : index
          scf.for %v_290 = %v_288 to %v_289 step %v_207 {
            %v_291 = arith.muli %v_218, %v_290 : index
            %v_292 = arith.addi %v_242, %v_291 : index
            %v_293 = arith.muli %v_243, %v_147 : index
            %v_294 = arith.addi %v_293, %v_290 : index
            %v_295 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_296 = builtin.unrealized_conversion_cast %v_295 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_297 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_298 = builtin.unrealized_conversion_cast %v_297 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_299 = memref.load %v_298[%v_292] : memref<?xf64>
            %v_300 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_301 = builtin.unrealized_conversion_cast %v_300 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_302 = memref.load %v_301[%v_273] : memref<?xf64>
            %v_303 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_304 = builtin.unrealized_conversion_cast %v_303 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_305 = memref.load %v_304[%v_245] : memref<?xf64>
            %v_306 = arith.mulf %v_302, %v_305 : f64
            %v_307 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_308 = builtin.unrealized_conversion_cast %v_307 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_309 = memref.load %v_308[%v_294] : memref<?xf64>
            %v_310 = arith.mulf %v_306, %v_309 : f64
            %v_311 = arith.addf %v_299, %v_310 : f64
            memref.store %v_311, %v_296[%v_292] : memref<?xf64>
          }
          %v_312 = arith.addi %v_207, %v_274 : index
          %v_313 = arith.addi %v_207, %v_273 : index
          %v_314 = arith.cmpi slt, %v_313, %v_252 : index
          %v_318 = scf.if %v_314 -> (index) {
            %v_315 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_316 = builtin.unrealized_conversion_cast %v_315 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_317 = memref.load %v_316[%v_313] : memref<?xindex>
            scf.yield %v_317 : index
          } else {
            scf.yield %v_27 : index
          }
          scf.yield %v_312, %v_313, %v_318 : index, index, index
        }
        %v_320 = arith.addi %v_207, %v_262 : index
        %v_321 = arith.maxsi %v_233, %v_320 : index
        scf.for %v_322 = %v_321 to %v_27 step %v_207 {
          %v_323 = arith.muli %v_218, %v_322 : index
          %v_324 = arith.addi %v_242, %v_323 : index
          %v_325 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_326 = builtin.unrealized_conversion_cast %v_325 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_327 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_328 = builtin.unrealized_conversion_cast %v_327 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_329 = memref.load %v_328[%v_324] : memref<?xf64>
          memref.store %v_329, %v_326[%v_324] : memref<?xf64>
        }
      }
    }
    %v_330 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_331 = builtin.unrealized_conversion_cast %v_330 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_332 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_333 = builtin.unrealized_conversion_cast %v_332 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_334 = memref.dim %v_333, %v_233 : memref<?xf64>
    %v_335 = arith.index_cast %v_334 : index to i64
    %v_336 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_335, %v_336 : i64, !llvm.ptr
    %v_337 = memref.extract_aligned_pointer_as_index %v_331 : memref<?xf64> -> index
    %v_338 = arith.index_cast %v_337 : index to i64
    %v_339 = llvm.inttoptr %v_338 : i64 to !llvm.ptr
    %v_340 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_339, %v_340 : !llvm.ptr, !llvm.ptr
    %v_341 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_342 = builtin.unrealized_conversion_cast %v_341 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_343 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_344 = builtin.unrealized_conversion_cast %v_343 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_345 = memref.dim %v_344, %v_233 : memref<?xindex>
    %v_346 = arith.index_cast %v_345 : index to i64
    %v_347 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_346, %v_347 : i64, !llvm.ptr
    %v_348 = memref.extract_aligned_pointer_as_index %v_342 : memref<?xindex> -> index
    %v_349 = arith.index_cast %v_348 : index to i64
    %v_350 = llvm.inttoptr %v_349 : i64 to !llvm.ptr
    %v_351 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_350, %v_351 : !llvm.ptr, !llvm.ptr
    %v_352 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_353 = builtin.unrealized_conversion_cast %v_352 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_354 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_355 = builtin.unrealized_conversion_cast %v_354 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_356 = memref.dim %v_355, %v_233 : memref<?xindex>
    %v_357 = arith.index_cast %v_356 : index to i64
    %v_358 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_357, %v_358 : i64, !llvm.ptr
    %v_359 = memref.extract_aligned_pointer_as_index %v_353 : memref<?xindex> -> index
    %v_360 = arith.index_cast %v_359 : index to i64
    %v_361 = llvm.inttoptr %v_360 : i64 to !llvm.ptr
    %v_362 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_361, %v_362 : !llvm.ptr, !llvm.ptr
    %v_363 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_364 = builtin.unrealized_conversion_cast %v_363 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_365 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_366 = builtin.unrealized_conversion_cast %v_365 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_367 = memref.dim %v_366, %v_233 : memref<?xf64>
    %v_368 = arith.index_cast %v_367 : index to i64
    %v_369 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_368, %v_369 : i64, !llvm.ptr
    %v_370 = memref.extract_aligned_pointer_as_index %v_364 : memref<?xf64> -> index
    %v_371 = arith.index_cast %v_370 : index to i64
    %v_372 = llvm.inttoptr %v_371 : i64 to !llvm.ptr
    %v_373 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_372, %v_373 : !llvm.ptr, !llvm.ptr
    %v_374 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_375 = builtin.unrealized_conversion_cast %v_374 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_376 = llvm.load %v_142 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_377 = builtin.unrealized_conversion_cast %v_376 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_378 = memref.dim %v_377, %v_233 : memref<?xf64>
    %v_379 = arith.index_cast %v_378 : index to i64
    %v_380 = llvm.getelementptr %v_129[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_379, %v_380 : i64, !llvm.ptr
    %v_381 = memref.extract_aligned_pointer_as_index %v_375 : memref<?xf64> -> index
    %v_382 = arith.index_cast %v_381 : index to i64
    %v_383 = llvm.inttoptr %v_382 : i64 to !llvm.ptr
    %v_384 = llvm.getelementptr %v_129[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_383, %v_384 : !llvm.ptr, !llvm.ptr
    %v_385 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_386 = builtin.unrealized_conversion_cast %v_385 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_387 = llvm.load %v_173 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_388 = builtin.unrealized_conversion_cast %v_387 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_389 = memref.dim %v_388, %v_233 : memref<?xf64>
    %v_390 = arith.index_cast %v_389 : index to i64
    %v_391 = llvm.getelementptr %v_160[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_390, %v_391 : i64, !llvm.ptr
    %v_392 = memref.extract_aligned_pointer_as_index %v_386 : memref<?xf64> -> index
    %v_393 = arith.index_cast %v_392 : index to i64
    %v_394 = llvm.inttoptr %v_393 : i64 to !llvm.ptr
    %v_395 = llvm.getelementptr %v_160[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_394, %v_395 : !llvm.ptr, !llvm.ptr
    %v_396 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_397 = llvm.insertvalue %_A_9, %v_396[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_397, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}

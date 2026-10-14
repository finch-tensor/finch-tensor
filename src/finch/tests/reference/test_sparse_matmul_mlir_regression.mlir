Compiling MLIR code:
module {
  func.func @scansearch_index(
    %arr: memref<?xindex>, %x: index,
    %lo: index, %hi: index
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

  func.func @main(%_A_15: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.ptr, %_A_7: !llvm.ptr, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
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
    %v_87 = llvm.extractvalue %_A_16[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, i64)>
    %v_88 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_89 = llvm.load %v_88 : !llvm.ptr -> !llvm.ptr
    %v_90 = llvm.getelementptr %v_89[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
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
    %v_109 = llvm.getelementptr %v_108[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_110 = llvm.load %v_109 : !llvm.ptr -> i64
    %v_111 = arith.index_cast %v_110 : i64 to index
    %v_112 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_113 = llvm.load %v_112 : !llvm.ptr -> !llvm.ptr
    %v_114 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_115 = llvm.load %v_114 : !llvm.ptr -> !llvm.ptr
    %v_116 = llvm.getelementptr %v_115[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_117 = llvm.load %v_116 : !llvm.ptr -> !llvm.ptr
    %v_118 = llvm.getelementptr %v_115[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_119 = llvm.load %v_118 : !llvm.ptr -> i64
    %v_120 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_121 = llvm.insertvalue %v_117, %v_120[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_122 = llvm.insertvalue %v_117, %v_121[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_123 = llvm.insertvalue %v_13, %v_122[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_124 = llvm.insertvalue %v_119, %v_123[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_125 = llvm.insertvalue %v_14, %v_124[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_126 = builtin.unrealized_conversion_cast %v_125 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_127 = builtin.unrealized_conversion_cast %v_126 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_128 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_127, %v_128 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_129 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_130 = llvm.load %v_129 : !llvm.ptr -> !llvm.ptr
    %v_131 = llvm.getelementptr %v_130[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_132 = llvm.load %v_131 : !llvm.ptr -> !llvm.ptr
    %v_133 = llvm.getelementptr %v_132[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_134 = llvm.load %v_133 : !llvm.ptr -> !llvm.ptr
    %v_135 = llvm.getelementptr %v_132[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_136 = llvm.load %v_135 : !llvm.ptr -> i64
    %v_137 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_138 = llvm.insertvalue %v_134, %v_137[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_139 = llvm.insertvalue %v_134, %v_138[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_140 = llvm.insertvalue %v_13, %v_139[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_141 = llvm.insertvalue %v_136, %v_140[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_142 = llvm.insertvalue %v_14, %v_141[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_143 = builtin.unrealized_conversion_cast %v_142 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_144 = builtin.unrealized_conversion_cast %v_143 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_145 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_144, %v_145 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_146 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_147 = llvm.load %v_146 : !llvm.ptr -> !llvm.ptr
    %v_148 = llvm.getelementptr %v_147[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_149 = llvm.load %v_148 : !llvm.ptr -> i64
    %v_150 = arith.index_cast %v_149 : i64 to index
    %v_151 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_152 = llvm.load %v_151 : !llvm.ptr -> !llvm.ptr
    %v_153 = llvm.getelementptr %v_152[0, 5] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_154 = llvm.load %v_153 : !llvm.ptr -> i64
    %v_155 = arith.index_cast %v_154 : i64 to index
    %v_156 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_157 = llvm.load %v_156 : !llvm.ptr -> !llvm.ptr
    %v_158 = llvm.getelementptr %v_157[0, 6] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_159 = llvm.load %v_158 : !llvm.ptr -> i64
    %v_160 = arith.index_cast %v_159 : i64 to index
    %v_161 = llvm.getelementptr %v_87[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_162 = llvm.load %v_161 : !llvm.ptr -> !llvm.ptr
    %v_163 = llvm.getelementptr %v_162[0, 7] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>
    %v_164 = llvm.load %v_163 : !llvm.ptr -> i1
    %v_165 = llvm.getelementptr %v_87[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_166 = llvm.load %v_165 : !llvm.ptr -> i64
    %v_167 = arith.index_cast %v_166 : i64 to index
    %v_168 = llvm.getelementptr %v_87[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, i64, i64)>
    %v_169 = llvm.load %v_168 : !llvm.ptr -> i64
    %v_170 = arith.index_cast %v_169 : i64 to index
    %v_171 = llvm.getelementptr %_A_18[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_172 = llvm.load %v_171 : !llvm.ptr -> !llvm.ptr
    %v_173 = llvm.getelementptr %v_172[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_174 = llvm.load %v_173 : !llvm.ptr -> !llvm.ptr
    %v_175 = llvm.getelementptr %v_172[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_176 = llvm.load %v_175 : !llvm.ptr -> i64
    %v_177 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_178 = llvm.insertvalue %v_174, %v_177[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_179 = llvm.insertvalue %v_174, %v_178[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_180 = llvm.insertvalue %v_13, %v_179[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_181 = llvm.insertvalue %v_176, %v_180[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_182 = llvm.insertvalue %v_14, %v_181[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_183 = builtin.unrealized_conversion_cast %v_182 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_184 = builtin.unrealized_conversion_cast %v_183 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_185 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_184, %v_185 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_186 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_187 = llvm.load %v_186 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_188 = llvm.extractvalue %v_187[0] : !llvm.struct<(i64, i64)>
    %v_189 = arith.index_cast %v_188 : i64 to index
    %v_190 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_191 = llvm.load %v_190 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_192 = llvm.extractvalue %v_191[1] : !llvm.struct<(i64, i64)>
    %v_193 = arith.index_cast %v_192 : i64 to index
    %v_194 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_195 = llvm.load %v_194 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_196 = llvm.extractvalue %v_195[0] : !llvm.struct<(i64, i64)>
    %v_197 = arith.index_cast %v_196 : i64 to index
    %v_198 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_199 = llvm.load %v_198 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_200 = llvm.extractvalue %v_199[1] : !llvm.struct<(i64, i64)>
    %v_201 = arith.index_cast %v_200 : i64 to index
    %v_202 = llvm.getelementptr %_A_7[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_203 = llvm.load %v_202 : !llvm.ptr -> !llvm.ptr
    %v_204 = llvm.getelementptr %v_203[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_205 = llvm.load %v_204 : !llvm.ptr -> !llvm.ptr
    %v_206 = llvm.getelementptr %v_203[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_207 = llvm.load %v_206 : !llvm.ptr -> i64
    %v_208 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_209 = llvm.insertvalue %v_205, %v_208[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_210 = llvm.insertvalue %v_205, %v_209[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_211 = llvm.insertvalue %v_13, %v_210[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_212 = llvm.insertvalue %v_207, %v_211[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_213 = llvm.insertvalue %v_14, %v_212[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_214 = builtin.unrealized_conversion_cast %v_213 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_215 = builtin.unrealized_conversion_cast %v_214 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_216 = llvm.alloca %v_14 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_215, %v_216 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_217 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_218 = llvm.load %v_217 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_219 = llvm.extractvalue %v_218[0] : !llvm.struct<(i64, i64)>
    %v_220 = arith.index_cast %v_219 : i64 to index
    %v_221 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_222 = llvm.load %v_221 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_223 = llvm.extractvalue %v_222[1] : !llvm.struct<(i64, i64)>
    %v_224 = arith.index_cast %v_223 : i64 to index
    %v_225 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_226 = llvm.load %v_225 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_227 = llvm.extractvalue %v_226[0] : !llvm.struct<(i64, i64)>
    %v_228 = arith.index_cast %v_227 : i64 to index
    %v_229 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_230 = llvm.load %v_229 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_231 = llvm.extractvalue %v_230[1] : !llvm.struct<(i64, i64)>
    %v_232 = arith.index_cast %v_231 : i64 to index
    %v_233 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_234 = arith.index_cast %v_27 : index to i64
    %v_235 = llvm.insertvalue %v_234, %v_233[0] : !llvm.struct<(i64, i64)>
    %v_236 = arith.index_cast %v_83 : index to i64
    %v_237 = llvm.insertvalue %v_236, %v_235[1] : !llvm.struct<(i64, i64)>
    %v_238 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_237, %v_238 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_239 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_240 = llvm.load %v_239 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_241 = llvm.extractvalue %v_240[0] : !llvm.struct<(i64, i64)>
    %v_242 = arith.index_cast %v_241 : i64 to index
    %v_243 = llvm.getelementptr %_A_18[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_244 = llvm.load %v_243 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_245 = llvm.extractvalue %v_244[1] : !llvm.struct<(i64, i64)>
    %v_246 = arith.index_cast %v_245 : i64 to index
    %v_247 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_248 = arith.index_cast %v_83 : index to i64
    %v_249 = llvm.insertvalue %v_248, %v_247[0] : !llvm.struct<(i64, i64)>
    %v_250 = arith.constant 1 : index
    %v_251 = arith.index_cast %v_250 : index to i64
    %v_252 = llvm.insertvalue %v_251, %v_249[1] : !llvm.struct<(i64, i64)>
    %v_253 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_252, %v_253 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_254 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_255 = llvm.load %v_254 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_256 = llvm.extractvalue %v_255[0] : !llvm.struct<(i64, i64)>
    %v_257 = arith.index_cast %v_256 : i64 to index
    %v_258 = llvm.getelementptr %_A_18[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_259 = llvm.load %v_258 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_260 = llvm.extractvalue %v_259[1] : !llvm.struct<(i64, i64)>
    %v_261 = arith.index_cast %v_260 : i64 to index
    %v_262 = arith.muli %v_27, %v_83 : index
    %v_263 = arith.index_cast %v_262 : index to i64
    %v_264 = llvm.getelementptr %v_172[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_265 = llvm.getelementptr %v_172[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_266 = llvm.load %v_265 : !llvm.ptr -> !llvm.ptr
    %v_267 = llvm.call %v_266(%v_264, %v_263) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_268 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_269 = llvm.insertvalue %v_267, %v_268[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_270 = llvm.insertvalue %v_267, %v_269[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_271 = llvm.insertvalue %v_13, %v_270[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_272 = llvm.insertvalue %v_263, %v_271[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_273 = llvm.insertvalue %v_14, %v_272[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_274 = builtin.unrealized_conversion_cast %v_273 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_275 = builtin.unrealized_conversion_cast %v_274 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %v_275, %v_185 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_276 = arith.constant 0 : index
    %v_277 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_278 = builtin.unrealized_conversion_cast %v_277 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_279 = memref.dim %v_278, %v_276 : memref<?xf64>
    scf.for %v_280 = %v_276 to %v_279 step %v_250 {
      %v_281 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_282 = builtin.unrealized_conversion_cast %v_281 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_283 = arith.constant 0.0 : f64
      memref.store %v_283, %v_282[%v_280] : memref<?xf64>
    }
    scf.for %v_284 = %v_276 to %v_27 step %v_250 {
      %v_285 = arith.muli %v_257, %v_284 : index
      scf.for %v_286 = %v_276 to %v_83 step %v_250 {
        %v_287 = arith.muli %v_261, %v_286 : index
        %v_288 = arith.addi %v_285, %v_287 : index
        %v_289 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_290 = builtin.unrealized_conversion_cast %v_289 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_291 = memref.load %v_290[%v_286] : memref<?xindex>
        %v_292 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_293 = builtin.unrealized_conversion_cast %v_292 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_294 = arith.addi %v_250, %v_286 : index
        %v_295 = memref.load %v_293[%v_294] : memref<?xindex>
        %v_296 = arith.cmpi slt, %v_291, %v_295 : index
        %v_304, %v_305 = scf.if %v_296 -> (index, index) {
          %v_297 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_298 = builtin.unrealized_conversion_cast %v_297 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_299 = memref.load %v_298[%v_291] : memref<?xindex>
          %v_300 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_301 = builtin.unrealized_conversion_cast %v_300 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_302 = arith.subi %v_295, %v_250 : index
          %v_303 = memref.load %v_301[%v_302] : memref<?xindex>
          scf.yield %v_299, %v_303 : index, index
        } else {
          scf.yield %v_250, %v_276 : index, index
        }
        %v_306 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_307 = builtin.unrealized_conversion_cast %v_306 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_308 = memref.load %v_307[%v_291] : memref<?xindex>
        %v_309 = arith.cmpi slt, %v_308, %v_276 : index
        %v_314 = scf.if %v_309 -> (index) {
          %v_310 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_311 = builtin.unrealized_conversion_cast %v_310 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_312 = arith.subi %v_295, %v_250 : index
          %v_313 = func.call @scansearch_index(%v_311, %v_276, %v_291, %v_312) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_313 : index
        } else {
          scf.yield %v_291 : index
        }
        %v_337:3 = scf.while (%v_315 = %v_276, %v_316 = %v_314, %v_317 = %v_304) : (index, index, index) -> (index, index, index) {
          %v_318 = arith.addi %v_250, %v_305 : index
          %v_319 = arith.minsi %v_27, %v_318 : index
          %v_320 = arith.cmpi slt, %v_317, %v_319 : index
          scf.condition(%v_320) %v_315, %v_316, %v_317 : index, index, index
        } do {
          ^bb(%v_315: index, %v_316: index, %v_317: index):
          %v_321 = arith.maxsi %v_315, %v_317 : index
          %v_322 = arith.addi %v_250, %v_317 : index
          scf.for %v_323 = %v_321 to %v_322 step %v_250 {
            %v_324 = arith.cmpi eq, %v_323, %v_284 : index
            scf.if %v_324 {
              %v_325 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_326 = builtin.unrealized_conversion_cast %v_325 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_327 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_328 = builtin.unrealized_conversion_cast %v_327 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_329 = memref.load %v_328[%v_316] : memref<?xf64>
              memref.store %v_329, %v_326[%v_288] : memref<?xf64>
            }
          }
          %v_330 = arith.addi %v_250, %v_317 : index
          %v_331 = arith.addi %v_250, %v_316 : index
          %v_332 = arith.cmpi slt, %v_331, %v_295 : index
          %v_336 = scf.if %v_332 -> (index) {
            %v_333 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_334 = builtin.unrealized_conversion_cast %v_333 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_335 = memref.load %v_334[%v_331] : memref<?xindex>
            scf.yield %v_335 : index
          } else {
            scf.yield %v_27 : index
          }
          scf.yield %v_330, %v_331, %v_336 : index, index, index
        }
      }
    }
    %v_338 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_339 = arith.index_cast %v_83 : index to i64
    %v_340 = llvm.insertvalue %v_339, %v_338[0] : !llvm.struct<(i64, i64)>
    %v_341 = arith.index_cast %v_111 : index to i64
    %v_342 = llvm.insertvalue %v_341, %v_340[1] : !llvm.struct<(i64, i64)>
    %v_343 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_342, %v_343 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_344 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_345 = llvm.load %v_344 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_346 = llvm.extractvalue %v_345[0] : !llvm.struct<(i64, i64)>
    %v_347 = arith.index_cast %v_346 : i64 to index
    %v_348 = llvm.getelementptr %_A_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_349 = llvm.load %v_348 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_350 = llvm.extractvalue %v_349[1] : !llvm.struct<(i64, i64)>
    %v_351 = arith.index_cast %v_350 : i64 to index
    %v_352 = llvm.mlir.undef : !llvm.struct<(i64, i64)>
    %v_353 = arith.index_cast %v_111 : index to i64
    %v_354 = llvm.insertvalue %v_353, %v_352[0] : !llvm.struct<(i64, i64)>
    %v_355 = arith.index_cast %v_250 : index to i64
    %v_356 = llvm.insertvalue %v_355, %v_354[1] : !llvm.struct<(i64, i64)>
    %v_357 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    llvm.store %v_356, %v_357 : !llvm.struct<(i64, i64)>, !llvm.ptr
    %v_358 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_359 = llvm.load %v_358 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_360 = llvm.extractvalue %v_359[0] : !llvm.struct<(i64, i64)>
    %v_361 = arith.index_cast %v_360 : i64 to index
    %v_362 = llvm.getelementptr %_A_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_363 = llvm.load %v_362 : !llvm.ptr -> !llvm.struct<(i64, i64)>
    %v_364 = llvm.extractvalue %v_363[1] : !llvm.struct<(i64, i64)>
    %v_365 = arith.index_cast %v_364 : i64 to index
    %v_366 = arith.muli %v_83, %v_111 : index
    %v_367 = arith.index_cast %v_366 : index to i64
    %v_368 = llvm.getelementptr %v_203[0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_369 = llvm.getelementptr %v_203[0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_370 = llvm.load %v_369 : !llvm.ptr -> !llvm.ptr
    %v_371 = llvm.call %v_370(%v_368, %v_367) : !llvm.ptr, (!llvm.ptr, i64) -> !llvm.ptr
    %v_372 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_373 = llvm.insertvalue %v_371, %v_372[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_374 = llvm.insertvalue %v_371, %v_373[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_375 = llvm.insertvalue %v_13, %v_374[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_376 = llvm.insertvalue %v_367, %v_375[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_377 = llvm.insertvalue %v_14, %v_376[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_378 = builtin.unrealized_conversion_cast %v_377 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_379 = builtin.unrealized_conversion_cast %v_378 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    llvm.store %v_379, %v_216 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_380 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_381 = builtin.unrealized_conversion_cast %v_380 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_382 = memref.dim %v_381, %v_276 : memref<?xf64>
    scf.for %v_383 = %v_276 to %v_382 step %v_250 {
      %v_384 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_385 = builtin.unrealized_conversion_cast %v_384 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_386 = arith.constant 0.0 : f64
      memref.store %v_386, %v_385[%v_383] : memref<?xf64>
    }
    scf.for %v_387 = %v_276 to %v_167 step %v_250 {
      %v_388 = arith.muli %v_257, %v_387 : index
      scf.for %v_389 = %v_276 to %v_83 step %v_250 {
        %v_390 = arith.muli %v_361, %v_389 : index
        %v_391 = arith.muli %v_261, %v_389 : index
        %v_392 = arith.addi %v_388, %v_391 : index
        %v_393 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_394 = builtin.unrealized_conversion_cast %v_393 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_395 = memref.load %v_394[%v_387] : memref<?xindex>
        %v_396 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_397 = builtin.unrealized_conversion_cast %v_396 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_398 = arith.addi %v_250, %v_387 : index
        %v_399 = memref.load %v_397[%v_398] : memref<?xindex>
        %v_400 = arith.cmpi slt, %v_395, %v_399 : index
        %v_408, %v_409 = scf.if %v_400 -> (index, index) {
          %v_401 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_402 = builtin.unrealized_conversion_cast %v_401 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_403 = memref.load %v_402[%v_395] : memref<?xindex>
          %v_404 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_405 = builtin.unrealized_conversion_cast %v_404 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_406 = arith.subi %v_399, %v_250 : index
          %v_407 = memref.load %v_405[%v_406] : memref<?xindex>
          scf.yield %v_403, %v_407 : index, index
        } else {
          scf.yield %v_250, %v_276 : index, index
        }
        %v_410 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_411 = builtin.unrealized_conversion_cast %v_410 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_412 = memref.load %v_411[%v_395] : memref<?xindex>
        %v_413 = arith.cmpi slt, %v_412, %v_276 : index
        %v_418 = scf.if %v_413 -> (index) {
          %v_414 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_415 = builtin.unrealized_conversion_cast %v_414 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_416 = arith.subi %v_399, %v_250 : index
          %v_417 = func.call @scansearch_index(%v_415, %v_276, %v_395, %v_416) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_417 : index
        } else {
          scf.yield %v_395 : index
        }
        %v_460:3 = scf.while (%v_419 = %v_276, %v_420 = %v_418, %v_421 = %v_408) : (index, index, index) -> (index, index, index) {
          %v_422 = arith.addi %v_250, %v_409 : index
          %v_423 = arith.minsi %v_111, %v_422 : index
          %v_424 = arith.cmpi slt, %v_421, %v_423 : index
          scf.condition(%v_424) %v_419, %v_420, %v_421 : index, index, index
        } do {
          ^bb_2(%v_419: index, %v_420: index, %v_421: index):
          %v_425 = arith.addi %v_250, %v_421 : index
          %v_426 = arith.minsi %v_425, %v_421 : index
          scf.for %v_427 = %v_419 to %v_426 step %v_250 {
            %v_428 = arith.muli %v_365, %v_427 : index
            %v_429 = arith.addi %v_390, %v_428 : index
            %v_430 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_431 = builtin.unrealized_conversion_cast %v_430 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_432 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_433 = builtin.unrealized_conversion_cast %v_432 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_434 = memref.load %v_433[%v_429] : memref<?xf64>
            memref.store %v_434, %v_431[%v_429] : memref<?xf64>
          }
          %v_435 = arith.maxsi %v_419, %v_421 : index
          %v_436 = arith.addi %v_250, %v_421 : index
          scf.for %v_437 = %v_435 to %v_436 step %v_250 {
            %v_438 = arith.muli %v_365, %v_437 : index
            %v_439 = arith.addi %v_390, %v_438 : index
            %v_440 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_441 = builtin.unrealized_conversion_cast %v_440 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_442 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_443 = builtin.unrealized_conversion_cast %v_442 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_444 = memref.load %v_443[%v_439] : memref<?xf64>
            %v_445 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_446 = builtin.unrealized_conversion_cast %v_445 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_447 = memref.load %v_446[%v_392] : memref<?xf64>
            %v_448 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_449 = builtin.unrealized_conversion_cast %v_448 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_450 = memref.load %v_449[%v_420] : memref<?xf64>
            %v_451 = arith.mulf %v_447, %v_450 : f64
            %v_452 = arith.addf %v_444, %v_451 : f64
            memref.store %v_452, %v_441[%v_439] : memref<?xf64>
          }
          %v_453 = arith.addi %v_250, %v_421 : index
          %v_454 = arith.addi %v_250, %v_420 : index
          %v_455 = arith.cmpi slt, %v_454, %v_399 : index
          %v_459 = scf.if %v_455 -> (index) {
            %v_456 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_457 = builtin.unrealized_conversion_cast %v_456 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_458 = memref.load %v_457[%v_454] : memref<?xindex>
            scf.yield %v_458 : index
          } else {
            scf.yield %v_111 : index
          }
          scf.yield %v_453, %v_454, %v_459 : index, index, index
        }
        %v_461 = arith.addi %v_250, %v_409 : index
        %v_462 = arith.maxsi %v_276, %v_461 : index
        scf.for %v_463 = %v_462 to %v_111 step %v_250 {
          %v_464 = arith.muli %v_365, %v_463 : index
          %v_465 = arith.addi %v_390, %v_464 : index
          %v_466 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_467 = builtin.unrealized_conversion_cast %v_466 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_468 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_469 = builtin.unrealized_conversion_cast %v_468 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_470 = memref.load %v_469[%v_465] : memref<?xf64>
          memref.store %v_470, %v_467[%v_465] : memref<?xf64>
        }
      }
    }
    %v_471 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_472 = builtin.unrealized_conversion_cast %v_471 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_473 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_474 = builtin.unrealized_conversion_cast %v_473 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_475 = memref.dim %v_474, %v_276 : memref<?xf64>
    %v_476 = arith.index_cast %v_475 : index to i64
    %v_477 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_476, %v_477 : i64, !llvm.ptr
    %v_478 = memref.extract_aligned_pointer_as_index %v_472 : memref<?xf64> -> index
    %v_479 = arith.index_cast %v_478 : index to i64
    %v_480 = llvm.inttoptr %v_479 : i64 to !llvm.ptr
    %v_481 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_480, %v_481 : !llvm.ptr, !llvm.ptr
    %v_482 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_483 = builtin.unrealized_conversion_cast %v_482 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_484 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_485 = builtin.unrealized_conversion_cast %v_484 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_486 = memref.dim %v_485, %v_276 : memref<?xindex>
    %v_487 = arith.index_cast %v_486 : index to i64
    %v_488 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_487, %v_488 : i64, !llvm.ptr
    %v_489 = memref.extract_aligned_pointer_as_index %v_483 : memref<?xindex> -> index
    %v_490 = arith.index_cast %v_489 : index to i64
    %v_491 = llvm.inttoptr %v_490 : i64 to !llvm.ptr
    %v_492 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_491, %v_492 : !llvm.ptr, !llvm.ptr
    %v_493 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_494 = builtin.unrealized_conversion_cast %v_493 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_495 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_496 = builtin.unrealized_conversion_cast %v_495 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_497 = memref.dim %v_496, %v_276 : memref<?xindex>
    %v_498 = arith.index_cast %v_497 : index to i64
    %v_499 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_498, %v_499 : i64, !llvm.ptr
    %v_500 = memref.extract_aligned_pointer_as_index %v_494 : memref<?xindex> -> index
    %v_501 = arith.index_cast %v_500 : index to i64
    %v_502 = llvm.inttoptr %v_501 : i64 to !llvm.ptr
    %v_503 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_502, %v_503 : !llvm.ptr, !llvm.ptr
    %v_504 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_505 = builtin.unrealized_conversion_cast %v_504 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_506 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_507 = builtin.unrealized_conversion_cast %v_506 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_508 = memref.dim %v_507, %v_276 : memref<?xf64>
    %v_509 = arith.index_cast %v_508 : index to i64
    %v_510 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_509, %v_510 : i64, !llvm.ptr
    %v_511 = memref.extract_aligned_pointer_as_index %v_505 : memref<?xf64> -> index
    %v_512 = arith.index_cast %v_511 : index to i64
    %v_513 = llvm.inttoptr %v_512 : i64 to !llvm.ptr
    %v_514 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_513, %v_514 : !llvm.ptr, !llvm.ptr
    %v_515 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_516 = builtin.unrealized_conversion_cast %v_515 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_517 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_518 = builtin.unrealized_conversion_cast %v_517 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_519 = memref.dim %v_518, %v_276 : memref<?xindex>
    %v_520 = arith.index_cast %v_519 : index to i64
    %v_521 = llvm.getelementptr %v_115[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_520, %v_521 : i64, !llvm.ptr
    %v_522 = memref.extract_aligned_pointer_as_index %v_516 : memref<?xindex> -> index
    %v_523 = arith.index_cast %v_522 : index to i64
    %v_524 = llvm.inttoptr %v_523 : i64 to !llvm.ptr
    %v_525 = llvm.getelementptr %v_115[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_524, %v_525 : !llvm.ptr, !llvm.ptr
    %v_526 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_527 = builtin.unrealized_conversion_cast %v_526 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_528 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_529 = builtin.unrealized_conversion_cast %v_528 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_530 = memref.dim %v_529, %v_276 : memref<?xindex>
    %v_531 = arith.index_cast %v_530 : index to i64
    %v_532 = llvm.getelementptr %v_132[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_531, %v_532 : i64, !llvm.ptr
    %v_533 = memref.extract_aligned_pointer_as_index %v_527 : memref<?xindex> -> index
    %v_534 = arith.index_cast %v_533 : index to i64
    %v_535 = llvm.inttoptr %v_534 : i64 to !llvm.ptr
    %v_536 = llvm.getelementptr %v_132[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_535, %v_536 : !llvm.ptr, !llvm.ptr
    %v_537 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_538 = builtin.unrealized_conversion_cast %v_537 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_539 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_540 = builtin.unrealized_conversion_cast %v_539 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_541 = memref.dim %v_540, %v_276 : memref<?xf64>
    %v_542 = arith.index_cast %v_541 : index to i64
    %v_543 = llvm.getelementptr %v_172[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_542, %v_543 : i64, !llvm.ptr
    %v_544 = memref.extract_aligned_pointer_as_index %v_538 : memref<?xf64> -> index
    %v_545 = arith.index_cast %v_544 : index to i64
    %v_546 = llvm.inttoptr %v_545 : i64 to !llvm.ptr
    %v_547 = llvm.getelementptr %v_172[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_546, %v_547 : !llvm.ptr, !llvm.ptr
    %v_548 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_549 = builtin.unrealized_conversion_cast %v_548 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_550 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_551 = builtin.unrealized_conversion_cast %v_550 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_552 = memref.dim %v_551, %v_276 : memref<?xf64>
    %v_553 = arith.index_cast %v_552 : index to i64
    %v_554 = llvm.getelementptr %v_203[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_553, %v_554 : i64, !llvm.ptr
    %v_555 = memref.extract_aligned_pointer_as_index %v_549 : memref<?xf64> -> index
    %v_556 = arith.index_cast %v_555 : index to i64
    %v_557 = llvm.inttoptr %v_556 : i64 to !llvm.ptr
    %v_558 = llvm.getelementptr %v_203[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_557, %v_558 : !llvm.ptr, !llvm.ptr
    %v_559 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_560 = llvm.insertvalue %_A_7, %v_559[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_560, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}

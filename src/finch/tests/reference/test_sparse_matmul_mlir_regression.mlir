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
    %v_233 = arith.constant 0 : index
    %v_234 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_235 = builtin.unrealized_conversion_cast %v_234 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_236 = memref.dim %v_235, %v_233 : memref<?xf64>
    %v_237 = arith.constant 1 : index
    scf.for %v_238 = %v_233 to %v_236 step %v_237 {
      %v_239 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_240 = builtin.unrealized_conversion_cast %v_239 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_241 = arith.constant 0.0 : f64
      memref.store %v_241, %v_240[%v_238] : memref<?xf64>
    }
    scf.for %v_242 = %v_233 to %v_27 step %v_237 {
      %v_243 = arith.muli %v_197, %v_242 : index
      scf.for %v_244 = %v_233 to %v_83 step %v_237 {
        %v_245 = arith.muli %v_201, %v_244 : index
        %v_246 = arith.addi %v_243, %v_245 : index
        %v_247 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_248 = builtin.unrealized_conversion_cast %v_247 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_249 = memref.load %v_248[%v_244] : memref<?xindex>
        %v_250 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_251 = builtin.unrealized_conversion_cast %v_250 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_252 = arith.addi %v_237, %v_244 : index
        %v_253 = memref.load %v_251[%v_252] : memref<?xindex>
        %v_254 = arith.cmpi slt, %v_249, %v_253 : index
        %v_262, %v_263 = scf.if %v_254 -> (index, index) {
          %v_255 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_256 = builtin.unrealized_conversion_cast %v_255 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_257 = memref.load %v_256[%v_249] : memref<?xindex>
          %v_258 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_259 = builtin.unrealized_conversion_cast %v_258 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_260 = arith.subi %v_253, %v_237 : index
          %v_261 = memref.load %v_259[%v_260] : memref<?xindex>
          scf.yield %v_257, %v_261 : index, index
        } else {
          scf.yield %v_237, %v_233 : index, index
        }
        %v_264 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_265 = builtin.unrealized_conversion_cast %v_264 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_266 = memref.load %v_265[%v_249] : memref<?xindex>
        %v_267 = arith.cmpi slt, %v_266, %v_233 : index
        %v_272 = scf.if %v_267 -> (index) {
          %v_268 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_269 = builtin.unrealized_conversion_cast %v_268 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_270 = arith.subi %v_253, %v_237 : index
          %v_271 = func.call @scansearch(%v_269, %v_233, %v_249, %v_270) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_271 : index
        } else {
          scf.yield %v_249 : index
        }
        %v_295:3 = scf.while (%v_273 = %v_233, %v_274 = %v_272, %v_275 = %v_262) : (index, index, index) -> (index, index, index) {
          %v_276 = arith.addi %v_237, %v_263 : index
          %v_277 = arith.minsi %v_27, %v_276 : index
          %v_278 = arith.cmpi slt, %v_275, %v_277 : index
          scf.condition(%v_278) %v_273, %v_274, %v_275 : index, index, index
        } do {
          ^bb(%v_273: index, %v_274: index, %v_275: index):
          %v_279 = arith.maxsi %v_273, %v_275 : index
          %v_280 = arith.addi %v_237, %v_275 : index
          scf.for %v_281 = %v_279 to %v_280 step %v_237 {
            %v_282 = arith.cmpi eq, %v_281, %v_242 : index
            scf.if %v_282 {
              %v_283 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_284 = builtin.unrealized_conversion_cast %v_283 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_285 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_286 = builtin.unrealized_conversion_cast %v_285 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_287 = memref.load %v_286[%v_274] : memref<?xf64>
              memref.store %v_287, %v_284[%v_246] : memref<?xf64>
            }
          }
          %v_288 = arith.addi %v_237, %v_275 : index
          %v_289 = arith.addi %v_237, %v_274 : index
          %v_290 = arith.cmpi slt, %v_289, %v_253 : index
          %v_294 = scf.if %v_290 -> (index) {
            %v_291 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_292 = builtin.unrealized_conversion_cast %v_291 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_293 = memref.load %v_292[%v_289] : memref<?xindex>
            scf.yield %v_293 : index
          } else {
            scf.yield %v_27 : index
          }
          scf.yield %v_288, %v_289, %v_294 : index, index, index
        }
      }
    }
    %v_296 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_297 = builtin.unrealized_conversion_cast %v_296 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_298 = memref.dim %v_297, %v_233 : memref<?xf64>
    scf.for %v_299 = %v_233 to %v_298 step %v_237 {
      %v_300 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_301 = builtin.unrealized_conversion_cast %v_300 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_302 = arith.constant 0.0 : f64
      memref.store %v_302, %v_301[%v_299] : memref<?xf64>
    }
    scf.for %v_303 = %v_233 to %v_167 step %v_237 {
      %v_304 = arith.muli %v_197, %v_303 : index
      scf.for %v_305 = %v_233 to %v_193 step %v_237 {
        %v_306 = arith.muli %v_228, %v_305 : index
        %v_307 = arith.muli %v_201, %v_305 : index
        %v_308 = arith.addi %v_304, %v_307 : index
        %v_309 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_310 = builtin.unrealized_conversion_cast %v_309 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_311 = memref.load %v_310[%v_303] : memref<?xindex>
        %v_312 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_313 = builtin.unrealized_conversion_cast %v_312 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_314 = arith.addi %v_237, %v_303 : index
        %v_315 = memref.load %v_313[%v_314] : memref<?xindex>
        %v_316 = arith.cmpi slt, %v_311, %v_315 : index
        %v_324, %v_325 = scf.if %v_316 -> (index, index) {
          %v_317 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_318 = builtin.unrealized_conversion_cast %v_317 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_319 = memref.load %v_318[%v_311] : memref<?xindex>
          %v_320 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_321 = builtin.unrealized_conversion_cast %v_320 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_322 = arith.subi %v_315, %v_237 : index
          %v_323 = memref.load %v_321[%v_322] : memref<?xindex>
          scf.yield %v_319, %v_323 : index, index
        } else {
          scf.yield %v_237, %v_233 : index, index
        }
        %v_326 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_327 = builtin.unrealized_conversion_cast %v_326 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_328 = memref.load %v_327[%v_311] : memref<?xindex>
        %v_329 = arith.cmpi slt, %v_328, %v_233 : index
        %v_334 = scf.if %v_329 -> (index) {
          %v_330 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_331 = builtin.unrealized_conversion_cast %v_330 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_332 = arith.subi %v_315, %v_237 : index
          %v_333 = func.call @scansearch(%v_331, %v_233, %v_311, %v_332) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_333 : index
        } else {
          scf.yield %v_311 : index
        }
        %v_376:3 = scf.while (%v_335 = %v_233, %v_336 = %v_334, %v_337 = %v_324) : (index, index, index) -> (index, index, index) {
          %v_338 = arith.addi %v_237, %v_325 : index
          %v_339 = arith.minsi %v_111, %v_338 : index
          %v_340 = arith.cmpi slt, %v_337, %v_339 : index
          scf.condition(%v_340) %v_335, %v_336, %v_337 : index, index, index
        } do {
          ^bb_2(%v_335: index, %v_336: index, %v_337: index):
          %v_341 = arith.addi %v_237, %v_337 : index
          %v_342 = arith.minsi %v_341, %v_337 : index
          scf.for %v_343 = %v_335 to %v_342 step %v_237 {
            %v_344 = arith.muli %v_232, %v_343 : index
            %v_345 = arith.addi %v_306, %v_344 : index
            %v_346 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_347 = builtin.unrealized_conversion_cast %v_346 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_348 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_349 = builtin.unrealized_conversion_cast %v_348 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_350 = memref.load %v_349[%v_345] : memref<?xf64>
            memref.store %v_350, %v_347[%v_345] : memref<?xf64>
          }
          %v_351 = arith.maxsi %v_335, %v_337 : index
          %v_352 = arith.addi %v_237, %v_337 : index
          scf.for %v_353 = %v_351 to %v_352 step %v_237 {
            %v_354 = arith.muli %v_232, %v_353 : index
            %v_355 = arith.addi %v_306, %v_354 : index
            %v_356 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_357 = builtin.unrealized_conversion_cast %v_356 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_358 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_359 = builtin.unrealized_conversion_cast %v_358 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_360 = memref.load %v_359[%v_355] : memref<?xf64>
            %v_361 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_362 = builtin.unrealized_conversion_cast %v_361 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_363 = memref.load %v_362[%v_308] : memref<?xf64>
            %v_364 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_365 = builtin.unrealized_conversion_cast %v_364 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_366 = memref.load %v_365[%v_336] : memref<?xf64>
            %v_367 = arith.mulf %v_363, %v_366 : f64
            %v_368 = arith.addf %v_360, %v_367 : f64
            memref.store %v_368, %v_357[%v_355] : memref<?xf64>
          }
          %v_369 = arith.addi %v_237, %v_337 : index
          %v_370 = arith.addi %v_237, %v_336 : index
          %v_371 = arith.cmpi slt, %v_370, %v_315 : index
          %v_375 = scf.if %v_371 -> (index) {
            %v_372 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_373 = builtin.unrealized_conversion_cast %v_372 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_374 = memref.load %v_373[%v_370] : memref<?xindex>
            scf.yield %v_374 : index
          } else {
            scf.yield %v_111 : index
          }
          scf.yield %v_369, %v_370, %v_375 : index, index, index
        }
        %v_377 = arith.addi %v_237, %v_325 : index
        %v_378 = arith.maxsi %v_233, %v_377 : index
        scf.for %v_379 = %v_378 to %v_111 step %v_237 {
          %v_380 = arith.muli %v_232, %v_379 : index
          %v_381 = arith.addi %v_306, %v_380 : index
          %v_382 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_383 = builtin.unrealized_conversion_cast %v_382 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_384 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_385 = builtin.unrealized_conversion_cast %v_384 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_386 = memref.load %v_385[%v_381] : memref<?xf64>
          memref.store %v_386, %v_383[%v_381] : memref<?xf64>
        }
      }
    }
    %v_387 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_388 = builtin.unrealized_conversion_cast %v_387 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_389 = llvm.load %v_22 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_390 = builtin.unrealized_conversion_cast %v_389 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_391 = memref.dim %v_390, %v_233 : memref<?xf64>
    %v_392 = arith.index_cast %v_391 : index to i64
    %v_393 = llvm.getelementptr %v_7[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_392, %v_393 : i64, !llvm.ptr
    %v_394 = memref.extract_aligned_pointer_as_index %v_388 : memref<?xf64> -> index
    %v_395 = arith.index_cast %v_394 : index to i64
    %v_396 = llvm.inttoptr %v_395 : i64 to !llvm.ptr
    %v_397 = llvm.getelementptr %v_7[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_396, %v_397 : !llvm.ptr, !llvm.ptr
    %v_398 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_399 = builtin.unrealized_conversion_cast %v_398 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_400 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_401 = builtin.unrealized_conversion_cast %v_400 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_402 = memref.dim %v_401, %v_233 : memref<?xindex>
    %v_403 = arith.index_cast %v_402 : index to i64
    %v_404 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_403, %v_404 : i64, !llvm.ptr
    %v_405 = memref.extract_aligned_pointer_as_index %v_399 : memref<?xindex> -> index
    %v_406 = arith.index_cast %v_405 : index to i64
    %v_407 = llvm.inttoptr %v_406 : i64 to !llvm.ptr
    %v_408 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_407, %v_408 : !llvm.ptr, !llvm.ptr
    %v_409 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_410 = builtin.unrealized_conversion_cast %v_409 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_411 = llvm.load %v_61 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_412 = builtin.unrealized_conversion_cast %v_411 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_413 = memref.dim %v_412, %v_233 : memref<?xindex>
    %v_414 = arith.index_cast %v_413 : index to i64
    %v_415 = llvm.getelementptr %v_48[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_414, %v_415 : i64, !llvm.ptr
    %v_416 = memref.extract_aligned_pointer_as_index %v_410 : memref<?xindex> -> index
    %v_417 = arith.index_cast %v_416 : index to i64
    %v_418 = llvm.inttoptr %v_417 : i64 to !llvm.ptr
    %v_419 = llvm.getelementptr %v_48[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_418, %v_419 : !llvm.ptr, !llvm.ptr
    %v_420 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_421 = builtin.unrealized_conversion_cast %v_420 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_422 = llvm.load %v_106 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_423 = builtin.unrealized_conversion_cast %v_422 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_424 = memref.dim %v_423, %v_233 : memref<?xf64>
    %v_425 = arith.index_cast %v_424 : index to i64
    %v_426 = llvm.getelementptr %v_93[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_425, %v_426 : i64, !llvm.ptr
    %v_427 = memref.extract_aligned_pointer_as_index %v_421 : memref<?xf64> -> index
    %v_428 = arith.index_cast %v_427 : index to i64
    %v_429 = llvm.inttoptr %v_428 : i64 to !llvm.ptr
    %v_430 = llvm.getelementptr %v_93[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_429, %v_430 : !llvm.ptr, !llvm.ptr
    %v_431 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_432 = builtin.unrealized_conversion_cast %v_431 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_433 = llvm.load %v_128 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_434 = builtin.unrealized_conversion_cast %v_433 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_435 = memref.dim %v_434, %v_233 : memref<?xindex>
    %v_436 = arith.index_cast %v_435 : index to i64
    %v_437 = llvm.getelementptr %v_115[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_436, %v_437 : i64, !llvm.ptr
    %v_438 = memref.extract_aligned_pointer_as_index %v_432 : memref<?xindex> -> index
    %v_439 = arith.index_cast %v_438 : index to i64
    %v_440 = llvm.inttoptr %v_439 : i64 to !llvm.ptr
    %v_441 = llvm.getelementptr %v_115[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_440, %v_441 : !llvm.ptr, !llvm.ptr
    %v_442 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_443 = builtin.unrealized_conversion_cast %v_442 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_444 = llvm.load %v_145 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_445 = builtin.unrealized_conversion_cast %v_444 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_446 = memref.dim %v_445, %v_233 : memref<?xindex>
    %v_447 = arith.index_cast %v_446 : index to i64
    %v_448 = llvm.getelementptr %v_132[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_447, %v_448 : i64, !llvm.ptr
    %v_449 = memref.extract_aligned_pointer_as_index %v_443 : memref<?xindex> -> index
    %v_450 = arith.index_cast %v_449 : index to i64
    %v_451 = llvm.inttoptr %v_450 : i64 to !llvm.ptr
    %v_452 = llvm.getelementptr %v_132[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_451, %v_452 : !llvm.ptr, !llvm.ptr
    %v_453 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_454 = builtin.unrealized_conversion_cast %v_453 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_455 = llvm.load %v_185 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_456 = builtin.unrealized_conversion_cast %v_455 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_457 = memref.dim %v_456, %v_233 : memref<?xf64>
    %v_458 = arith.index_cast %v_457 : index to i64
    %v_459 = llvm.getelementptr %v_172[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_458, %v_459 : i64, !llvm.ptr
    %v_460 = memref.extract_aligned_pointer_as_index %v_454 : memref<?xf64> -> index
    %v_461 = arith.index_cast %v_460 : index to i64
    %v_462 = llvm.inttoptr %v_461 : i64 to !llvm.ptr
    %v_463 = llvm.getelementptr %v_172[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_462, %v_463 : !llvm.ptr, !llvm.ptr
    %v_464 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_465 = builtin.unrealized_conversion_cast %v_464 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_466 = llvm.load %v_216 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_467 = builtin.unrealized_conversion_cast %v_466 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_468 = memref.dim %v_467, %v_233 : memref<?xf64>
    %v_469 = arith.index_cast %v_468 : index to i64
    %v_470 = llvm.getelementptr %v_203[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_469, %v_470 : i64, !llvm.ptr
    %v_471 = memref.extract_aligned_pointer_as_index %v_465 : memref<?xf64> -> index
    %v_472 = arith.index_cast %v_471 : index to i64
    %v_473 = llvm.inttoptr %v_472 : i64 to !llvm.ptr
    %v_474 = llvm.getelementptr %v_203[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_473, %v_474 : !llvm.ptr, !llvm.ptr
    %v_475 = llvm.mlir.undef : !llvm.struct<(!llvm.ptr)>
    %v_476 = llvm.insertvalue %_A_7, %v_475[0] : !llvm.struct<(!llvm.ptr)>
    llvm.store %v_476, %_ret : !llvm.struct<(!llvm.ptr)>, !llvm.ptr
    func.return
  }
}

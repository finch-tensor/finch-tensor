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

  func.func @main(%_A_15: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_16: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>, %_A_18: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_A_7: !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_15[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_2 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_3 = llvm.load %v_2 : !llvm.ptr -> !llvm.ptr
    %v_4 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_5 = llvm.load %v_4 : !llvm.ptr -> i64
    %v_6 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_7 = arith.constant 0 : i64
    %v_8 = arith.constant 1 : i64
    %v_9 = llvm.insertvalue %v_3, %v_6[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_10 = llvm.insertvalue %v_3, %v_9[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_11 = llvm.insertvalue %v_7, %v_10[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_12 = llvm.insertvalue %v_5, %v_11[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_13 = llvm.insertvalue %v_8, %v_12[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_14 = builtin.unrealized_conversion_cast %v_13 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_15 = builtin.unrealized_conversion_cast %v_14 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_16 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_15, %v_16 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_17 = llvm.extractvalue %_A_15[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_18 = llvm.getelementptr %v_17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_19 = llvm.load %v_18 : !llvm.ptr -> !llvm.ptr
    %v_20 = llvm.getelementptr %v_17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_21 = llvm.load %v_20 : !llvm.ptr -> i64
    %v_22 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_23 = llvm.insertvalue %v_19, %v_22[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_24 = llvm.insertvalue %v_19, %v_23[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_25 = llvm.insertvalue %v_7, %v_24[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_26 = llvm.insertvalue %v_21, %v_25[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_27 = llvm.insertvalue %v_8, %v_26[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_28 = builtin.unrealized_conversion_cast %v_27 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_29 = builtin.unrealized_conversion_cast %v_28 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_30 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_29, %v_30 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_31 = llvm.extractvalue %_A_15[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_32 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_33 = llvm.load %v_32 : !llvm.ptr -> !llvm.ptr
    %v_34 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_35 = llvm.load %v_34 : !llvm.ptr -> i64
    %v_36 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_37 = llvm.insertvalue %v_33, %v_36[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_38 = llvm.insertvalue %v_33, %v_37[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_39 = llvm.insertvalue %v_7, %v_38[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_40 = llvm.insertvalue %v_35, %v_39[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_41 = llvm.insertvalue %v_8, %v_40[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_42 = builtin.unrealized_conversion_cast %v_41 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_43 = builtin.unrealized_conversion_cast %v_42 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_44 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_43, %v_44 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_45 = llvm.extractvalue %_A_15[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_46 = arith.index_cast %v_45 : i64 to index
    %v_47 = llvm.extractvalue %_A_15[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_48 = arith.index_cast %v_47 : i64 to index
    %v_49 = llvm.extractvalue %_A_16[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_50 = llvm.getelementptr %v_49[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_51 = llvm.load %v_50 : !llvm.ptr -> !llvm.ptr
    %v_52 = llvm.getelementptr %v_49[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_53 = llvm.load %v_52 : !llvm.ptr -> i64
    %v_54 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_55 = llvm.insertvalue %v_51, %v_54[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_56 = llvm.insertvalue %v_51, %v_55[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_57 = llvm.insertvalue %v_7, %v_56[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_58 = llvm.insertvalue %v_53, %v_57[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_59 = llvm.insertvalue %v_8, %v_58[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_60 = builtin.unrealized_conversion_cast %v_59 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_61 = builtin.unrealized_conversion_cast %v_60 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_62 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_61, %v_62 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_63 = llvm.extractvalue %_A_16[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_64 = llvm.getelementptr %v_63[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_65 = llvm.load %v_64 : !llvm.ptr -> !llvm.ptr
    %v_66 = llvm.getelementptr %v_63[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_67 = llvm.load %v_66 : !llvm.ptr -> i64
    %v_68 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_69 = llvm.insertvalue %v_65, %v_68[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_70 = llvm.insertvalue %v_65, %v_69[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_71 = llvm.insertvalue %v_7, %v_70[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_72 = llvm.insertvalue %v_67, %v_71[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_73 = llvm.insertvalue %v_8, %v_72[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_74 = builtin.unrealized_conversion_cast %v_73 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_75 = builtin.unrealized_conversion_cast %v_74 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_76 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_75, %v_76 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_77 = llvm.extractvalue %_A_16[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_78 = llvm.getelementptr %v_77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_79 = llvm.load %v_78 : !llvm.ptr -> !llvm.ptr
    %v_80 = llvm.getelementptr %v_77[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_81 = llvm.load %v_80 : !llvm.ptr -> i64
    %v_82 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_83 = llvm.insertvalue %v_79, %v_82[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_84 = llvm.insertvalue %v_79, %v_83[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_85 = llvm.insertvalue %v_7, %v_84[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_86 = llvm.insertvalue %v_81, %v_85[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_87 = llvm.insertvalue %v_8, %v_86[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_88 = builtin.unrealized_conversion_cast %v_87 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_89 = builtin.unrealized_conversion_cast %v_88 : memref<?xindex> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_90 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_89, %v_90 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_91 = llvm.extractvalue %_A_16[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_92 = arith.index_cast %v_91 : i64 to index
    %v_93 = llvm.extractvalue %_A_16[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.ptr)>, i64, !llvm.ptr, !llvm.ptr, i64, i64, i64, i1)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64)>
    %v_94 = arith.index_cast %v_93 : i64 to index
    %v_95 = llvm.extractvalue %_A_18[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_96 = llvm.getelementptr %v_95[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_97 = llvm.load %v_96 : !llvm.ptr -> !llvm.ptr
    %v_98 = llvm.getelementptr %v_95[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_99 = llvm.load %v_98 : !llvm.ptr -> i64
    %v_100 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_101 = llvm.insertvalue %v_97, %v_100[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_102 = llvm.insertvalue %v_97, %v_101[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_103 = llvm.insertvalue %v_7, %v_102[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_104 = llvm.insertvalue %v_99, %v_103[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_105 = llvm.insertvalue %v_8, %v_104[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_106 = builtin.unrealized_conversion_cast %v_105 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_107 = builtin.unrealized_conversion_cast %v_106 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_108 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_107, %v_108 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_109 = llvm.extractvalue %_A_18[1, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_110 = arith.index_cast %v_109 : i64 to index
    %v_111 = llvm.extractvalue %_A_18[1, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_112 = arith.index_cast %v_111 : i64 to index
    %v_113 = llvm.extractvalue %_A_7[0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_114 = llvm.getelementptr %v_113[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_115 = llvm.load %v_114 : !llvm.ptr -> !llvm.ptr
    %v_116 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    %v_117 = llvm.load %v_116 : !llvm.ptr -> i64
    %v_118 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_119 = llvm.insertvalue %v_115, %v_118[0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_120 = llvm.insertvalue %v_115, %v_119[1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_121 = llvm.insertvalue %v_7, %v_120[2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_122 = llvm.insertvalue %v_117, %v_121[3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_123 = llvm.insertvalue %v_8, %v_122[4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_124 = builtin.unrealized_conversion_cast %v_123 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_125 = builtin.unrealized_conversion_cast %v_124 : memref<?xf64> to !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_126 = llvm.alloca %v_8 x !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> : (i64) -> !llvm.ptr
    llvm.store %v_125, %v_126 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.ptr
    %v_127 = llvm.extractvalue %_A_7[1, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_128 = arith.index_cast %v_127 : i64 to index
    %v_129 = llvm.extractvalue %_A_7[1, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_130 = arith.index_cast %v_129 : i64 to index
    %v_131 = arith.constant 0 : index
    %v_132 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_133 = builtin.unrealized_conversion_cast %v_132 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_134 = memref.dim %v_133, %v_131 : memref<?xf64>
    %v_135 = arith.constant 1 : index
    scf.for %v_136 = %v_131 to %v_134 step %v_135 {
      %v_137 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_138 = builtin.unrealized_conversion_cast %v_137 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_139 = arith.constant 0.0 : f64
      memref.store %v_139, %v_138[%v_136] : memref<?xf64>
    }
    scf.for %v_140 = %v_131 to %v_48 step %v_135 {
      %v_141 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_142 = arith.index_cast %v_141 : i64 to index
      %v_143 = arith.muli %v_142, %v_140 : index
      scf.for %v_144 = %v_131 to %v_46 step %v_135 {
        %v_145 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_146 = arith.index_cast %v_145 : i64 to index
        %v_147 = arith.muli %v_146, %v_144 : index
        %v_148 = arith.addi %v_143, %v_147 : index
        %v_149 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_150 = builtin.unrealized_conversion_cast %v_149 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_151 = memref.load %v_150[%v_144] : memref<?xindex>
        %v_152 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_153 = builtin.unrealized_conversion_cast %v_152 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_154 = arith.addi %v_135, %v_144 : index
        %v_155 = memref.load %v_153[%v_154] : memref<?xindex>
        %v_156 = arith.cmpi slt, %v_151, %v_155 : index
        %v_164, %v_165 = scf.if %v_156 -> (index, index) {
          %v_157 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_158 = builtin.unrealized_conversion_cast %v_157 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_159 = memref.load %v_158[%v_151] : memref<?xindex>
          %v_160 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_161 = builtin.unrealized_conversion_cast %v_160 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_162 = arith.subi %v_155, %v_135 : index
          %v_163 = memref.load %v_161[%v_162] : memref<?xindex>
          scf.yield %v_159, %v_163 : index, index
        } else {
          scf.yield %v_135, %v_131 : index, index
        }
        %v_166 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_167 = builtin.unrealized_conversion_cast %v_166 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_168 = memref.load %v_167[%v_151] : memref<?xindex>
        %v_169 = arith.cmpi slt, %v_168, %v_131 : index
        %v_174 = scf.if %v_169 -> (index) {
          %v_170 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_171 = builtin.unrealized_conversion_cast %v_170 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_172 = arith.subi %v_155, %v_135 : index
          %v_173 = func.call @scansearch_index(%v_171, %v_131, %v_151, %v_172) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_173 : index
        } else {
          scf.yield %v_151 : index
        }
        %v_197:3 = scf.while (%v_175 = %v_131, %v_176 = %v_174, %v_177 = %v_164) : (index, index, index) -> (index, index, index) {
          %v_178 = arith.addi %v_135, %v_165 : index
          %v_179 = arith.minsi %v_48, %v_178 : index
          %v_180 = arith.cmpi slt, %v_177, %v_179 : index
          scf.condition(%v_180) %v_175, %v_176, %v_177 : index, index, index
        } do {
          ^bb(%v_175: index, %v_176: index, %v_177: index):
          %v_181 = arith.maxsi %v_175, %v_177 : index
          %v_182 = arith.addi %v_135, %v_177 : index
          scf.for %v_183 = %v_181 to %v_182 step %v_135 {
            %v_184 = arith.cmpi eq, %v_183, %v_140 : index
            scf.if %v_184 {
              %v_185 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_186 = builtin.unrealized_conversion_cast %v_185 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_187 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_188 = builtin.unrealized_conversion_cast %v_187 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_189 = memref.load %v_188[%v_176] : memref<?xf64>
              memref.store %v_189, %v_186[%v_148] : memref<?xf64>
            }
          }
          %v_190 = arith.addi %v_135, %v_177 : index
          %v_191 = arith.addi %v_135, %v_176 : index
          %v_192 = arith.cmpi slt, %v_191, %v_155 : index
          %v_196 = scf.if %v_192 -> (index) {
            %v_193 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_194 = builtin.unrealized_conversion_cast %v_193 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_195 = memref.load %v_194[%v_191] : memref<?xindex>
            scf.yield %v_195 : index
          } else {
            scf.yield %v_48 : index
          }
          scf.yield %v_190, %v_191, %v_196 : index, index, index
        }
      }
    }
    %v_198 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_199 = builtin.unrealized_conversion_cast %v_198 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_200 = memref.dim %v_199, %v_131 : memref<?xf64>
    scf.for %v_201 = %v_131 to %v_200 step %v_135 {
      %v_202 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_203 = builtin.unrealized_conversion_cast %v_202 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_204 = arith.constant 0.0 : f64
      memref.store %v_204, %v_203[%v_201] : memref<?xf64>
    }
    scf.for %v_205 = %v_131 to %v_92 step %v_135 {
      %v_206 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_207 = arith.index_cast %v_206 : i64 to index
      %v_208 = arith.muli %v_207, %v_205 : index
      scf.for %v_209 = %v_131 to %v_112 step %v_135 {
        %v_210 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_211 = arith.index_cast %v_210 : i64 to index
        %v_212 = arith.muli %v_211, %v_209 : index
        %v_213 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_214 = arith.index_cast %v_213 : i64 to index
        %v_215 = arith.muli %v_214, %v_209 : index
        %v_216 = arith.addi %v_208, %v_215 : index
        %v_217 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_218 = builtin.unrealized_conversion_cast %v_217 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_219 = memref.load %v_218[%v_205] : memref<?xindex>
        %v_220 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_221 = builtin.unrealized_conversion_cast %v_220 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_222 = arith.addi %v_135, %v_205 : index
        %v_223 = memref.load %v_221[%v_222] : memref<?xindex>
        %v_224 = arith.cmpi slt, %v_219, %v_223 : index
        %v_232, %v_233 = scf.if %v_224 -> (index, index) {
          %v_225 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_226 = builtin.unrealized_conversion_cast %v_225 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_227 = memref.load %v_226[%v_219] : memref<?xindex>
          %v_228 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_229 = builtin.unrealized_conversion_cast %v_228 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_230 = arith.subi %v_223, %v_135 : index
          %v_231 = memref.load %v_229[%v_230] : memref<?xindex>
          scf.yield %v_227, %v_231 : index, index
        } else {
          scf.yield %v_135, %v_131 : index, index
        }
        %v_234 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_235 = builtin.unrealized_conversion_cast %v_234 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_236 = memref.load %v_235[%v_219] : memref<?xindex>
        %v_237 = arith.cmpi slt, %v_236, %v_131 : index
        %v_242 = scf.if %v_237 -> (index) {
          %v_238 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_239 = builtin.unrealized_conversion_cast %v_238 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_240 = arith.subi %v_223, %v_135 : index
          %v_241 = func.call @scansearch_index(%v_239, %v_131, %v_219, %v_240) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_241 : index
        } else {
          scf.yield %v_219 : index
        }
        %v_288:3 = scf.while (%v_243 = %v_131, %v_244 = %v_242, %v_245 = %v_232) : (index, index, index) -> (index, index, index) {
          %v_246 = arith.addi %v_135, %v_233 : index
          %v_247 = arith.minsi %v_94, %v_246 : index
          %v_248 = arith.cmpi slt, %v_245, %v_247 : index
          scf.condition(%v_248) %v_243, %v_244, %v_245 : index, index, index
        } do {
          ^bb_2(%v_243: index, %v_244: index, %v_245: index):
          %v_249 = arith.addi %v_135, %v_245 : index
          %v_250 = arith.minsi %v_249, %v_245 : index
          scf.for %v_251 = %v_243 to %v_250 step %v_135 {
            %v_252 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_253 = arith.index_cast %v_252 : i64 to index
            %v_254 = arith.muli %v_253, %v_251 : index
            %v_255 = arith.addi %v_212, %v_254 : index
            %v_256 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_257 = builtin.unrealized_conversion_cast %v_256 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_258 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_259 = builtin.unrealized_conversion_cast %v_258 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_260 = memref.load %v_259[%v_255] : memref<?xf64>
            memref.store %v_260, %v_257[%v_255] : memref<?xf64>
          }
          %v_261 = arith.maxsi %v_243, %v_245 : index
          %v_262 = arith.addi %v_135, %v_245 : index
          scf.for %v_263 = %v_261 to %v_262 step %v_135 {
            %v_264 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_265 = arith.index_cast %v_264 : i64 to index
            %v_266 = arith.muli %v_265, %v_263 : index
            %v_267 = arith.addi %v_212, %v_266 : index
            %v_268 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_269 = builtin.unrealized_conversion_cast %v_268 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_270 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_271 = builtin.unrealized_conversion_cast %v_270 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_272 = memref.load %v_271[%v_267] : memref<?xf64>
            %v_273 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_274 = builtin.unrealized_conversion_cast %v_273 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_275 = memref.load %v_274[%v_216] : memref<?xf64>
            %v_276 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_277 = builtin.unrealized_conversion_cast %v_276 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_278 = memref.load %v_277[%v_244] : memref<?xf64>
            %v_279 = arith.mulf %v_275, %v_278 : f64
            %v_280 = arith.addf %v_272, %v_279 : f64
            memref.store %v_280, %v_269[%v_267] : memref<?xf64>
          }
          %v_281 = arith.addi %v_135, %v_245 : index
          %v_282 = arith.addi %v_135, %v_244 : index
          %v_283 = arith.cmpi slt, %v_282, %v_223 : index
          %v_287 = scf.if %v_283 -> (index) {
            %v_284 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_285 = builtin.unrealized_conversion_cast %v_284 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_286 = memref.load %v_285[%v_282] : memref<?xindex>
            scf.yield %v_286 : index
          } else {
            scf.yield %v_94 : index
          }
          scf.yield %v_281, %v_282, %v_287 : index, index, index
        }
        %v_289 = arith.addi %v_135, %v_233 : index
        %v_290 = arith.maxsi %v_131, %v_289 : index
        scf.for %v_291 = %v_290 to %v_94 step %v_135 {
          %v_292 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_293 = arith.index_cast %v_292 : i64 to index
          %v_294 = arith.muli %v_293, %v_291 : index
          %v_295 = arith.addi %v_212, %v_294 : index
          %v_296 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_297 = builtin.unrealized_conversion_cast %v_296 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_298 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_299 = builtin.unrealized_conversion_cast %v_298 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_300 = memref.load %v_299[%v_295] : memref<?xf64>
          memref.store %v_300, %v_297[%v_295] : memref<?xf64>
        }
      }
    }
    %v_301 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_302 = builtin.unrealized_conversion_cast %v_301 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_303 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_304 = builtin.unrealized_conversion_cast %v_303 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_305 = memref.dim %v_304, %v_131 : memref<?xf64>
    %v_306 = arith.index_cast %v_305 : index to i64
    %v_307 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_306, %v_307 : i64, !llvm.ptr
    %v_308 = memref.extract_aligned_pointer_as_index %v_302 : memref<?xf64> -> index
    %v_309 = arith.index_cast %v_308 : index to i64
    %v_310 = llvm.inttoptr %v_309 : i64 to !llvm.ptr
    %v_311 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_310, %v_311 : !llvm.ptr, !llvm.ptr
    %v_312 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_313 = builtin.unrealized_conversion_cast %v_312 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_314 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_315 = builtin.unrealized_conversion_cast %v_314 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_316 = memref.dim %v_315, %v_131 : memref<?xindex>
    %v_317 = arith.index_cast %v_316 : index to i64
    %v_318 = llvm.getelementptr %v_17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_317, %v_318 : i64, !llvm.ptr
    %v_319 = memref.extract_aligned_pointer_as_index %v_313 : memref<?xindex> -> index
    %v_320 = arith.index_cast %v_319 : index to i64
    %v_321 = llvm.inttoptr %v_320 : i64 to !llvm.ptr
    %v_322 = llvm.getelementptr %v_17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_321, %v_322 : !llvm.ptr, !llvm.ptr
    %v_323 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_324 = builtin.unrealized_conversion_cast %v_323 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_325 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_326 = builtin.unrealized_conversion_cast %v_325 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_327 = memref.dim %v_326, %v_131 : memref<?xindex>
    %v_328 = arith.index_cast %v_327 : index to i64
    %v_329 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_328, %v_329 : i64, !llvm.ptr
    %v_330 = memref.extract_aligned_pointer_as_index %v_324 : memref<?xindex> -> index
    %v_331 = arith.index_cast %v_330 : index to i64
    %v_332 = llvm.inttoptr %v_331 : i64 to !llvm.ptr
    %v_333 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_332, %v_333 : !llvm.ptr, !llvm.ptr
    %v_334 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_335 = builtin.unrealized_conversion_cast %v_334 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_336 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_337 = builtin.unrealized_conversion_cast %v_336 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_338 = memref.dim %v_337, %v_131 : memref<?xf64>
    %v_339 = arith.index_cast %v_338 : index to i64
    %v_340 = llvm.getelementptr %v_49[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_339, %v_340 : i64, !llvm.ptr
    %v_341 = memref.extract_aligned_pointer_as_index %v_335 : memref<?xf64> -> index
    %v_342 = arith.index_cast %v_341 : index to i64
    %v_343 = llvm.inttoptr %v_342 : i64 to !llvm.ptr
    %v_344 = llvm.getelementptr %v_49[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_343, %v_344 : !llvm.ptr, !llvm.ptr
    %v_345 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_346 = builtin.unrealized_conversion_cast %v_345 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_347 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_348 = builtin.unrealized_conversion_cast %v_347 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_349 = memref.dim %v_348, %v_131 : memref<?xindex>
    %v_350 = arith.index_cast %v_349 : index to i64
    %v_351 = llvm.getelementptr %v_63[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_350, %v_351 : i64, !llvm.ptr
    %v_352 = memref.extract_aligned_pointer_as_index %v_346 : memref<?xindex> -> index
    %v_353 = arith.index_cast %v_352 : index to i64
    %v_354 = llvm.inttoptr %v_353 : i64 to !llvm.ptr
    %v_355 = llvm.getelementptr %v_63[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_354, %v_355 : !llvm.ptr, !llvm.ptr
    %v_356 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_357 = builtin.unrealized_conversion_cast %v_356 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_358 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_359 = builtin.unrealized_conversion_cast %v_358 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_360 = memref.dim %v_359, %v_131 : memref<?xindex>
    %v_361 = arith.index_cast %v_360 : index to i64
    %v_362 = llvm.getelementptr %v_77[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_361, %v_362 : i64, !llvm.ptr
    %v_363 = memref.extract_aligned_pointer_as_index %v_357 : memref<?xindex> -> index
    %v_364 = arith.index_cast %v_363 : index to i64
    %v_365 = llvm.inttoptr %v_364 : i64 to !llvm.ptr
    %v_366 = llvm.getelementptr %v_77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_365, %v_366 : !llvm.ptr, !llvm.ptr
    %v_367 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_368 = builtin.unrealized_conversion_cast %v_367 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_369 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_370 = builtin.unrealized_conversion_cast %v_369 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_371 = memref.dim %v_370, %v_131 : memref<?xf64>
    %v_372 = arith.index_cast %v_371 : index to i64
    %v_373 = llvm.getelementptr %v_95[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_372, %v_373 : i64, !llvm.ptr
    %v_374 = memref.extract_aligned_pointer_as_index %v_368 : memref<?xf64> -> index
    %v_375 = arith.index_cast %v_374 : index to i64
    %v_376 = llvm.inttoptr %v_375 : i64 to !llvm.ptr
    %v_377 = llvm.getelementptr %v_95[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_376, %v_377 : !llvm.ptr, !llvm.ptr
    %v_378 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_379 = builtin.unrealized_conversion_cast %v_378 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_380 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_381 = builtin.unrealized_conversion_cast %v_380 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_382 = memref.dim %v_381, %v_131 : memref<?xf64>
    %v_383 = arith.index_cast %v_382 : index to i64
    %v_384 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_383, %v_384 : i64, !llvm.ptr
    %v_385 = memref.extract_aligned_pointer_as_index %v_379 : memref<?xf64> -> index
    %v_386 = arith.index_cast %v_385 : index to i64
    %v_387 = llvm.inttoptr %v_386 : i64 to !llvm.ptr
    %v_388 = llvm.getelementptr %v_113[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_387, %v_388 : !llvm.ptr, !llvm.ptr
    %v_389 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_390 = llvm.insertvalue %_A_7, %v_389[0] : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_390, %_ret : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}

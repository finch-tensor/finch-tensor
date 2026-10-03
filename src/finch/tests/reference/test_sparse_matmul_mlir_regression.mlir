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
    scf.for %v_140 = %v_131 to %v_46 step %v_135 {
      %v_141 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_142 = builtin.unrealized_conversion_cast %v_141 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
      %v_143 = memref.load %v_142[%v_140] : memref<?xindex>
      %v_144 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_145 = builtin.unrealized_conversion_cast %v_144 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
      %v_146 = arith.addi %v_135, %v_140 : index
      %v_147 = memref.load %v_145[%v_146] : memref<?xindex>
      %v_148 = arith.cmpi slt, %v_143, %v_147 : index
      %v_156, %v_157 = scf.if %v_148 -> (index, index) {
        %v_149 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_150 = builtin.unrealized_conversion_cast %v_149 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_151 = memref.load %v_150[%v_143] : memref<?xindex>
        %v_152 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_153 = builtin.unrealized_conversion_cast %v_152 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_154 = arith.subi %v_147, %v_135 : index
        %v_155 = memref.load %v_153[%v_154] : memref<?xindex>
        scf.yield %v_151, %v_155 : index, index
      } else {
        scf.yield %v_135, %v_131 : index, index
      }
      %v_158 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_159 = builtin.unrealized_conversion_cast %v_158 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
      %v_160 = memref.load %v_159[%v_143] : memref<?xindex>
      %v_161 = arith.cmpi slt, %v_160, %v_131 : index
      %v_166 = scf.if %v_161 -> (index) {
        %v_162 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_163 = builtin.unrealized_conversion_cast %v_162 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_164 = arith.subi %v_147, %v_135 : index
        %v_165 = func.call @scansearch(%v_163, %v_131, %v_143, %v_164) : (memref<?xindex>, index, index, index) -> index
        scf.yield %v_165 : index
      } else {
        scf.yield %v_143 : index
      }
      %v_208:3 = scf.while (%v_167 = %v_131, %v_168 = %v_166, %v_169 = %v_156) : (index, index, index) -> (index, index, index) {
        %v_170 = arith.addi %v_135, %v_157 : index
        %v_171 = arith.minsi %v_48, %v_170 : index
        %v_172 = arith.cmpi slt, %v_169, %v_171 : index
        scf.condition(%v_172) %v_167, %v_168, %v_169 : index, index, index
      } do {
        ^bb(%v_167: index, %v_168: index, %v_169: index):
        %v_173 = arith.addi %v_135, %v_169 : index
        %v_174 = arith.minsi %v_173, %v_169 : index
        scf.for %v_175 = %v_167 to %v_174 step %v_135 {
          %v_176 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_177 = arith.index_cast %v_176 : i64 to index
          %v_178 = arith.muli %v_177, %v_175 : index
          scf.for %v_179 = %v_131 to %v_46 step %v_135 {
            %v_180 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_181 = arith.index_cast %v_180 : i64 to index
            %v_182 = arith.muli %v_181, %v_179 : index
            %v_183 = arith.addi %v_178, %v_182 : index
          }
        }
        %v_184 = arith.maxsi %v_167, %v_169 : index
        %v_185 = arith.addi %v_135, %v_169 : index
        scf.for %v_186 = %v_184 to %v_185 step %v_135 {
          %v_187 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_188 = arith.index_cast %v_187 : i64 to index
          %v_189 = arith.muli %v_188, %v_186 : index
          scf.for %v_190 = %v_131 to %v_46 step %v_135 {
            %v_191 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_192 = arith.index_cast %v_191 : i64 to index
            %v_193 = arith.muli %v_192, %v_190 : index
            %v_194 = arith.addi %v_189, %v_193 : index
            %v_195 = arith.cmpi eq, %v_190, %v_140 : index
            scf.if %v_195 {
              %v_196 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_197 = builtin.unrealized_conversion_cast %v_196 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_198 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
              %v_199 = builtin.unrealized_conversion_cast %v_198 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
              %v_200 = memref.load %v_199[%v_168] : memref<?xf64>
              memref.store %v_200, %v_197[%v_194] : memref<?xf64>
            }
          }
        }
        %v_201 = arith.addi %v_135, %v_169 : index
        %v_202 = arith.addi %v_135, %v_168 : index
        %v_203 = arith.cmpi slt, %v_202, %v_147 : index
        %v_207 = scf.if %v_203 -> (index) {
          %v_204 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_205 = builtin.unrealized_conversion_cast %v_204 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_206 = memref.load %v_205[%v_202] : memref<?xindex>
          scf.yield %v_206 : index
        } else {
          scf.yield %v_48 : index
        }
        scf.yield %v_201, %v_202, %v_207 : index, index, index
      }
      %v_209 = arith.addi %v_135, %v_157 : index
      %v_210 = arith.maxsi %v_131, %v_209 : index
      scf.for %v_211 = %v_210 to %v_48 step %v_135 {
        %v_212 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_213 = arith.index_cast %v_212 : i64 to index
        %v_214 = arith.muli %v_213, %v_211 : index
        scf.for %v_215 = %v_131 to %v_46 step %v_135 {
          %v_216 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_217 = arith.index_cast %v_216 : i64 to index
          %v_218 = arith.muli %v_217, %v_215 : index
          %v_219 = arith.addi %v_214, %v_218 : index
        }
      }
    }
    %v_220 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_221 = builtin.unrealized_conversion_cast %v_220 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_222 = memref.dim %v_221, %v_131 : memref<?xf64>
    scf.for %v_223 = %v_131 to %v_222 step %v_135 {
      %v_224 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
      %v_225 = builtin.unrealized_conversion_cast %v_224 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
      %v_226 = arith.constant 0.0 : f64
      memref.store %v_226, %v_225[%v_223] : memref<?xf64>
    }
    scf.for %v_227 = %v_131 to %v_92 step %v_135 {
      %v_228 = llvm.extractvalue %_A_18[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_229 = arith.index_cast %v_228 : i64 to index
      %v_230 = arith.muli %v_229, %v_227 : index
      scf.for %v_231 = %v_131 to %v_112 step %v_135 {
        %v_232 = llvm.extractvalue %_A_7[2, 0] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_233 = arith.index_cast %v_232 : i64 to index
        %v_234 = arith.muli %v_233, %v_231 : index
        %v_235 = llvm.extractvalue %_A_18[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
        %v_236 = arith.index_cast %v_235 : i64 to index
        %v_237 = arith.muli %v_236, %v_231 : index
        %v_238 = arith.addi %v_230, %v_237 : index
        %v_239 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_240 = builtin.unrealized_conversion_cast %v_239 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_241 = memref.load %v_240[%v_227] : memref<?xindex>
        %v_242 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_243 = builtin.unrealized_conversion_cast %v_242 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_244 = arith.addi %v_135, %v_227 : index
        %v_245 = memref.load %v_243[%v_244] : memref<?xindex>
        %v_246 = arith.cmpi slt, %v_241, %v_245 : index
        %v_254, %v_255 = scf.if %v_246 -> (index, index) {
          %v_247 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_248 = builtin.unrealized_conversion_cast %v_247 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_249 = memref.load %v_248[%v_241] : memref<?xindex>
          %v_250 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_251 = builtin.unrealized_conversion_cast %v_250 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_252 = arith.subi %v_245, %v_135 : index
          %v_253 = memref.load %v_251[%v_252] : memref<?xindex>
          scf.yield %v_249, %v_253 : index, index
        } else {
          scf.yield %v_135, %v_131 : index, index
        }
        %v_256 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
        %v_257 = builtin.unrealized_conversion_cast %v_256 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
        %v_258 = memref.load %v_257[%v_241] : memref<?xindex>
        %v_259 = arith.cmpi slt, %v_258, %v_131 : index
        %v_264 = scf.if %v_259 -> (index) {
          %v_260 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_261 = builtin.unrealized_conversion_cast %v_260 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
          %v_262 = arith.subi %v_245, %v_135 : index
          %v_263 = func.call @scansearch(%v_261, %v_131, %v_241, %v_262) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_263 : index
        } else {
          scf.yield %v_241 : index
        }
        %v_310:3 = scf.while (%v_265 = %v_131, %v_266 = %v_264, %v_267 = %v_254) : (index, index, index) -> (index, index, index) {
          %v_268 = arith.addi %v_135, %v_255 : index
          %v_269 = arith.minsi %v_94, %v_268 : index
          %v_270 = arith.cmpi slt, %v_267, %v_269 : index
          scf.condition(%v_270) %v_265, %v_266, %v_267 : index, index, index
        } do {
          ^bb_2(%v_265: index, %v_266: index, %v_267: index):
          %v_271 = arith.addi %v_135, %v_267 : index
          %v_272 = arith.minsi %v_271, %v_267 : index
          scf.for %v_273 = %v_265 to %v_272 step %v_135 {
            %v_274 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_275 = arith.index_cast %v_274 : i64 to index
            %v_276 = arith.muli %v_275, %v_273 : index
            %v_277 = arith.addi %v_234, %v_276 : index
            %v_278 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_279 = builtin.unrealized_conversion_cast %v_278 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_280 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_281 = builtin.unrealized_conversion_cast %v_280 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_282 = memref.load %v_281[%v_277] : memref<?xf64>
            memref.store %v_282, %v_279[%v_277] : memref<?xf64>
          }
          %v_283 = arith.maxsi %v_265, %v_267 : index
          %v_284 = arith.addi %v_135, %v_267 : index
          scf.for %v_285 = %v_283 to %v_284 step %v_135 {
            %v_286 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_287 = arith.index_cast %v_286 : i64 to index
            %v_288 = arith.muli %v_287, %v_285 : index
            %v_289 = arith.addi %v_234, %v_288 : index
            %v_290 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_291 = builtin.unrealized_conversion_cast %v_290 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_292 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_293 = builtin.unrealized_conversion_cast %v_292 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_294 = memref.load %v_293[%v_289] : memref<?xf64>
            %v_295 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_296 = builtin.unrealized_conversion_cast %v_295 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_297 = memref.load %v_296[%v_238] : memref<?xf64>
            %v_298 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_299 = builtin.unrealized_conversion_cast %v_298 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
            %v_300 = memref.load %v_299[%v_266] : memref<?xf64>
            %v_301 = arith.mulf %v_297, %v_300 : f64
            %v_302 = arith.addf %v_294, %v_301 : f64
            memref.store %v_302, %v_291[%v_289] : memref<?xf64>
          }
          %v_303 = arith.addi %v_135, %v_267 : index
          %v_304 = arith.addi %v_135, %v_266 : index
          %v_305 = arith.cmpi slt, %v_304, %v_245 : index
          %v_309 = scf.if %v_305 -> (index) {
            %v_306 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
            %v_307 = builtin.unrealized_conversion_cast %v_306 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
            %v_308 = memref.load %v_307[%v_304] : memref<?xindex>
            scf.yield %v_308 : index
          } else {
            scf.yield %v_94 : index
          }
          scf.yield %v_303, %v_304, %v_309 : index, index, index
        }
        %v_311 = arith.addi %v_135, %v_255 : index
        %v_312 = arith.maxsi %v_131, %v_311 : index
        scf.for %v_313 = %v_312 to %v_94 step %v_135 {
          %v_314 = llvm.extractvalue %_A_7[2, 1] : !llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_315 = arith.index_cast %v_314 : i64 to index
          %v_316 = arith.muli %v_315, %v_313 : index
          %v_317 = arith.addi %v_234, %v_316 : index
          %v_318 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_319 = builtin.unrealized_conversion_cast %v_318 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_320 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
          %v_321 = builtin.unrealized_conversion_cast %v_320 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
          %v_322 = memref.load %v_321[%v_317] : memref<?xf64>
          memref.store %v_322, %v_319[%v_317] : memref<?xf64>
        }
      }
    }
    %v_323 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_324 = builtin.unrealized_conversion_cast %v_323 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_325 = llvm.load %v_16 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_326 = builtin.unrealized_conversion_cast %v_325 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_327 = memref.dim %v_326, %v_131 : memref<?xf64>
    %v_328 = arith.index_cast %v_327 : index to i64
    %v_329 = llvm.getelementptr %v[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_328, %v_329 : i64, !llvm.ptr
    %v_330 = memref.extract_aligned_pointer_as_index %v_324 : memref<?xf64> -> index
    %v_331 = arith.index_cast %v_330 : index to i64
    %v_332 = llvm.inttoptr %v_331 : i64 to !llvm.ptr
    %v_333 = llvm.getelementptr %v[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_332, %v_333 : !llvm.ptr, !llvm.ptr
    %v_334 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_335 = builtin.unrealized_conversion_cast %v_334 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_336 = llvm.load %v_30 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_337 = builtin.unrealized_conversion_cast %v_336 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_338 = memref.dim %v_337, %v_131 : memref<?xindex>
    %v_339 = arith.index_cast %v_338 : index to i64
    %v_340 = llvm.getelementptr %v_17[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_339, %v_340 : i64, !llvm.ptr
    %v_341 = memref.extract_aligned_pointer_as_index %v_335 : memref<?xindex> -> index
    %v_342 = arith.index_cast %v_341 : index to i64
    %v_343 = llvm.inttoptr %v_342 : i64 to !llvm.ptr
    %v_344 = llvm.getelementptr %v_17[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_343, %v_344 : !llvm.ptr, !llvm.ptr
    %v_345 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_346 = builtin.unrealized_conversion_cast %v_345 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_347 = llvm.load %v_44 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_348 = builtin.unrealized_conversion_cast %v_347 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_349 = memref.dim %v_348, %v_131 : memref<?xindex>
    %v_350 = arith.index_cast %v_349 : index to i64
    %v_351 = llvm.getelementptr %v_31[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_350, %v_351 : i64, !llvm.ptr
    %v_352 = memref.extract_aligned_pointer_as_index %v_346 : memref<?xindex> -> index
    %v_353 = arith.index_cast %v_352 : index to i64
    %v_354 = llvm.inttoptr %v_353 : i64 to !llvm.ptr
    %v_355 = llvm.getelementptr %v_31[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_354, %v_355 : !llvm.ptr, !llvm.ptr
    %v_356 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_357 = builtin.unrealized_conversion_cast %v_356 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_358 = llvm.load %v_62 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_359 = builtin.unrealized_conversion_cast %v_358 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_360 = memref.dim %v_359, %v_131 : memref<?xf64>
    %v_361 = arith.index_cast %v_360 : index to i64
    %v_362 = llvm.getelementptr %v_49[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_361, %v_362 : i64, !llvm.ptr
    %v_363 = memref.extract_aligned_pointer_as_index %v_357 : memref<?xf64> -> index
    %v_364 = arith.index_cast %v_363 : index to i64
    %v_365 = llvm.inttoptr %v_364 : i64 to !llvm.ptr
    %v_366 = llvm.getelementptr %v_49[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_365, %v_366 : !llvm.ptr, !llvm.ptr
    %v_367 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_368 = builtin.unrealized_conversion_cast %v_367 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_369 = llvm.load %v_76 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_370 = builtin.unrealized_conversion_cast %v_369 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_371 = memref.dim %v_370, %v_131 : memref<?xindex>
    %v_372 = arith.index_cast %v_371 : index to i64
    %v_373 = llvm.getelementptr %v_63[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_372, %v_373 : i64, !llvm.ptr
    %v_374 = memref.extract_aligned_pointer_as_index %v_368 : memref<?xindex> -> index
    %v_375 = arith.index_cast %v_374 : index to i64
    %v_376 = llvm.inttoptr %v_375 : i64 to !llvm.ptr
    %v_377 = llvm.getelementptr %v_63[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_376, %v_377 : !llvm.ptr, !llvm.ptr
    %v_378 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_379 = builtin.unrealized_conversion_cast %v_378 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_380 = llvm.load %v_90 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_381 = builtin.unrealized_conversion_cast %v_380 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_382 = memref.dim %v_381, %v_131 : memref<?xindex>
    %v_383 = arith.index_cast %v_382 : index to i64
    %v_384 = llvm.getelementptr %v_77[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_383, %v_384 : i64, !llvm.ptr
    %v_385 = memref.extract_aligned_pointer_as_index %v_379 : memref<?xindex> -> index
    %v_386 = arith.index_cast %v_385 : index to i64
    %v_387 = llvm.inttoptr %v_386 : i64 to !llvm.ptr
    %v_388 = llvm.getelementptr %v_77[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_387, %v_388 : !llvm.ptr, !llvm.ptr
    %v_389 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_390 = builtin.unrealized_conversion_cast %v_389 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_391 = llvm.load %v_108 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_392 = builtin.unrealized_conversion_cast %v_391 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_393 = memref.dim %v_392, %v_131 : memref<?xf64>
    %v_394 = arith.index_cast %v_393 : index to i64
    %v_395 = llvm.getelementptr %v_95[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_394, %v_395 : i64, !llvm.ptr
    %v_396 = memref.extract_aligned_pointer_as_index %v_390 : memref<?xf64> -> index
    %v_397 = arith.index_cast %v_396 : index to i64
    %v_398 = llvm.inttoptr %v_397 : i64 to !llvm.ptr
    %v_399 = llvm.getelementptr %v_95[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_398, %v_399 : !llvm.ptr, !llvm.ptr
    %v_400 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_401 = builtin.unrealized_conversion_cast %v_400 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_402 = llvm.load %v_126 : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    %v_403 = builtin.unrealized_conversion_cast %v_402 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_404 = memref.dim %v_403, %v_131 : memref<?xf64>
    %v_405 = arith.index_cast %v_404 : index to i64
    %v_406 = llvm.getelementptr %v_113[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_405, %v_406 : i64, !llvm.ptr
    %v_407 = memref.extract_aligned_pointer_as_index %v_401 : memref<?xf64> -> index
    %v_408 = arith.index_cast %v_407 : index to i64
    %v_409 = llvm.inttoptr %v_408 : i64 to !llvm.ptr
    %v_410 = llvm.getelementptr %v_113[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr)>
    llvm.store %v_409, %v_410 : !llvm.ptr, !llvm.ptr
    %v_411 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_412 = llvm.insertvalue %_A_7, %v_411[0] : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_412, %_ret : !llvm.struct<(!llvm.struct<(!llvm.ptr, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}

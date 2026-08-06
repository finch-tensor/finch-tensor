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

  func.func @main(%_A_19: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_20: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %_A_21: !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>, %__A: !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>, %_ret: !llvm.ptr) attributes {llvm.emit_c_interface} {
    %v = llvm.extractvalue %_A_19[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_2 = builtin.unrealized_conversion_cast %v : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_3 = llvm.extractvalue %_A_19[0, 0, 3] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_4 = builtin.unrealized_conversion_cast %v_3 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_5 = llvm.extractvalue %_A_19[0, 0, 4] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_6 = builtin.unrealized_conversion_cast %v_5 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xindex>
    %v_7 = llvm.extractvalue %_A_19[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_8 = arith.index_cast %v_7 : i64 to index
    %v_9 = llvm.extractvalue %_A_19[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_10 = arith.index_cast %v_9 : i64 to index
    %v_11 = llvm.extractvalue %_A_20[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_12 = builtin.unrealized_conversion_cast %v_11 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_13 = llvm.extractvalue %_A_20[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_14 = arith.index_cast %v_13 : i64 to index
    %v_15 = llvm.extractvalue %_A_20[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_16 = arith.index_cast %v_15 : i64 to index
    %v_17 = llvm.extractvalue %_A_21[0, 0, 0, 0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_18 = builtin.unrealized_conversion_cast %v_17 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_19 = llvm.extractvalue %_A_21[0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_20 = arith.index_cast %v_19 : i64 to index
    %v_21 = llvm.extractvalue %_A_21[0, 0, 1] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
    %v_22 = arith.index_cast %v_21 : i64 to index
    %v_23 = llvm.extractvalue %__A[0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_24 = builtin.unrealized_conversion_cast %v_23 : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> to memref<?xf64>
    %v_25 = llvm.extractvalue %__A[1, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_26 = arith.index_cast %v_25 : i64 to index
    %v_27 = llvm.extractvalue %__A[1, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
    %v_28 = arith.index_cast %v_27 : i64 to index
    %v_29 = arith.constant 0 : index
    %v_30 = memref.dim %v_24, %v_29 : memref<?xf64>
    %v_31 = arith.constant 1 : index
    scf.for %v_32 = %v_29 to %v_30 step %v_31 {
      %v_33 = arith.constant 0.0 : f64
      memref.store %v_33, %v_24[%v_32] : memref<?xf64>
    }
    scf.for %v_34 = %v_29 to %v_8 step %v_31 {
      %v_35 = llvm.extractvalue %__A[2, 0] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
      %v_36 = arith.index_cast %v_35 : i64 to index
      %v_37 = arith.muli %v_36, %v_34 : index
      %v_38 = arith.addi %v_29, %v_37 : index
      %v_39 = llvm.extractvalue %_A_19[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_40 = arith.index_cast %v_39 : i64 to index
      %v_41 = arith.muli %v_40, %v_34 : index
      %v_42 = arith.addi %v_29, %v_41 : index
      %v_43 = llvm.extractvalue %_A_20[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
      %v_44 = arith.index_cast %v_43 : i64 to index
      %v_45 = arith.muli %v_44, %v_34 : index
      %v_46 = arith.addi %v_29, %v_45 : index
      scf.for %v_47 = %v_29 to %v_16 step %v_31 {
        %v_48 = llvm.extractvalue %_A_20[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_49 = arith.index_cast %v_48 : i64 to index
        %v_50 = arith.muli %v_49, %v_47 : index
        %v_51 = arith.addi %v_46, %v_50 : index
        %v_52 = llvm.extractvalue %_A_21[0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
        %v_53 = arith.index_cast %v_52 : i64 to index
        %v_54 = arith.muli %v_53, %v_47 : index
        %v_55 = arith.addi %v_29, %v_54 : index
        %v_56 = memref.load %v_4[%v_42] : memref<?xindex>
        %v_57 = arith.addi %v_42, %v_31 : index
        %v_58 = memref.load %v_4[%v_57] : memref<?xindex>
        %v_59 = arith.cmpi slt, %v_56, %v_58 : index
        %v_63, %v_64 = scf.if %v_59 -> (index, index) {
          %v_60 = memref.load %v_6[%v_56] : memref<?xindex>
          %v_61 = arith.subi %v_58, %v_31 : index
          %v_62 = memref.load %v_6[%v_61] : memref<?xindex>
          scf.yield %v_62, %v_60 : index, index
        } else {
          scf.yield %v_29, %v_31 : index, index
        }
        %v_65 = memref.load %v_6[%v_56] : memref<?xindex>
        %v_66 = arith.cmpi slt, %v_65, %v_29 : index
        %v_69 = scf.if %v_66 -> (index) {
          %v_67 = arith.subi %v_58, %v_31 : index
          %v_68 = func.call @scansearch(%v_6, %v_29, %v_56, %v_67) : (memref<?xindex>, index, index, index) -> index
          scf.yield %v_68 : index
        } else {
          scf.yield %v_56 : index
        }
        %v_106:3 = scf.while (%v_70 = %v_29, %v_71 = %v_69, %v_72 = %v_64) : (index, index, index) -> (index, index, index) {
          %v_73 = arith.addi %v_63, %v_31 : index
          %v_74 = arith.minsi %v_10, %v_73 : index
          %v_75 = arith.cmpi slt, %v_72, %v_74 : index
          scf.condition(%v_75) %v_70, %v_71, %v_72 : index, index, index
        } do {
          ^bb(%v_70: index, %v_71: index, %v_72: index):
          %v_76 = arith.addi %v_72, %v_31 : index
          %v_77 = arith.minsi %v_76, %v_72 : index
          scf.for %v_78 = %v_70 to %v_77 step %v_31 {
            %v_79 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_80 = arith.index_cast %v_79 : i64 to index
            %v_81 = arith.muli %v_80, %v_78 : index
            %v_82 = arith.addi %v_38, %v_81 : index
          }
          %v_83 = arith.maxsi %v_70, %v_72 : index
          %v_84 = arith.addi %v_72, %v_31 : index
          scf.for %v_85 = %v_83 to %v_84 step %v_31 {
            %v_86 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
            %v_87 = arith.index_cast %v_86 : i64 to index
            %v_88 = arith.muli %v_87, %v_85 : index
            %v_89 = arith.addi %v_38, %v_88 : index
            %v_90 = llvm.extractvalue %_A_21[0, 0, 2] : !llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>)>, i64, i64)>, i64, i64)>, !llvm.struct<(i64, i64)>, i64, i1)>
            %v_91 = arith.index_cast %v_90 : i64 to index
            %v_92 = arith.muli %v_91, %v_85 : index
            %v_93 = arith.addi %v_55, %v_92 : index
            %v_94 = memref.load %v_24[%v_89] : memref<?xf64>
            %v_95 = memref.load %v_2[%v_71] : memref<?xf64>
            %v_96 = memref.load %v_12[%v_51] : memref<?xf64>
            %v_97 = memref.load %v_18[%v_93] : memref<?xf64>
            %v_98 = arith.mulf %v_96, %v_97 : f64
            %v_99 = arith.mulf %v_95, %v_98 : f64
            %v_100 = arith.addf %v_94, %v_99 : f64
            memref.store %v_100, %v_24[%v_89] : memref<?xf64>
          }
          %v_101 = arith.addi %v_72, %v_31 : index
          %v_102 = arith.addi %v_71, %v_31 : index
          %v_103 = arith.cmpi slt, %v_102, %v_58 : index
          %v_105 = scf.if %v_103 -> (index) {
            %v_104 = memref.load %v_6[%v_102] : memref<?xindex>
            scf.yield %v_104 : index
          } else {
            scf.yield %v_10 : index
          }
          scf.yield %v_101, %v_102, %v_105 : index, index, index
        }
        %v_107 = arith.addi %v_63, %v_31 : index
        %v_108 = arith.maxsi %v_29, %v_107 : index
        scf.for %v_109 = %v_108 to %v_10 step %v_31 {
          %v_110 = llvm.extractvalue %__A[2, 1] : !llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>
          %v_111 = arith.index_cast %v_110 : i64 to index
          %v_112 = arith.muli %v_111, %v_109 : index
          %v_113 = arith.addi %v_38, %v_112 : index
        }
      }
    }
    %v_114 = llvm.mlir.undef : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    %v_115 = llvm.insertvalue %__A, %v_114[0] : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>
    llvm.store %v_115, %_ret : !llvm.struct<(!llvm.struct<(!llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>, !llvm.struct<(i64, i64)>, !llvm.struct<(i64, i64)>)>)>, !llvm.ptr
    func.return
  }
}

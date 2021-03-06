using Plots
gr(size=(600,400))
default(fmt = :png)

function separa_matriz(A,i,j)
    # cria matriz B com dados de A da coluna i até j-1 e cria matriz C com o resto

    lin=size(A)[1]
    col=size(A)[2]
    B=zeros(lin,j-i)
    C=zeros(lin,col-(j-i))

    for k=i:j-1
        B[:,k-i+1]=A[:,k]
    end
    for k=1:i-1
        C[:,k]=A[:,k]
    end
    for k=j:col
        C[:,k-(j-i)]=A[:,k]
    end
    return B, C
end

function main()
    A=readcsv("dados.csv")
    x=A[:,1]
    y=A[:,2]

    kfold(x,y)

    p = 8 ####### Sua escolha
    xlin = linspace(extrema(x)..., 100)
    β = reg_poly(x, y, p)
    ylin = β[1] * ones(100)
    for j = 1:p
        ylin .+= β[j+1] * xlin.^j
    end
    scatter(x, y, ms=3, c=:blue)
    plot!(xlin, ylin, c=:red, lw=2)
    png("ajuste")

    m=length(x)
    y_pred=zeros(1,m)
    for i=1:m
        y_pred[1,i]= β[1]
        for j=2:p+1
            y_pred[1,i] = y_pred[1,i]+β[j]*x[i]^(j-1)
        end
    end

    y_med = mean(y)
    soma1=0;
    soma2=0;

    for i=1:m
        soma1 = soma1 + (y_pred[1,i]-y_med)^2
        soma2 = soma2 + (y[i]-y_pred[1,i])^2
    end

    R2 = 1- soma2/soma1

    println("R2 = ", R2)

end

function reg_poly(x, y, p)
    m = length(x)
    A = [ones(m) [x[i]^j for i = 1:m, j = 1:p]]
    β = (A' * A) \ (A' * y)
    return β
end

function kfold(x,y; num_folds=5, max_p = 15)
    m=length(x)
    I=randperm(m)

    #tamanho medio de cada fold
    tam_conj = div(m,num_folds)
    r=m%num_folds
    dados_emb=zeros(2,m)

    #embaralhando os dados:
    for i=1:m
        dados_emb[1,i]=x[I[i]]
        dados_emb[2,i]=y[I[i]]
    end
    Erro_TR=zeros(num_folds, max_p)
    Erro_TE=zeros(num_folds, max_p)

    #pequeno malabarismo para tratar o caso em que o número de pontos não é divisível pelo número de folds: fizemos quantos folds
    #foram possíveis com um elemento a mais que os outros
    for i=1:r

        #separa os dados de teste e treinamento para cada fold usando a função separa_matriz

        Teste = separa_matriz(dados_emb,1+(i-1)*(tam_conj+1),1+i*(tam_conj+1))[1]
        Treinamento = separa_matriz(dados_emb,1+(i-1)*(tam_conj+1),1+i*(tam_conj+1))[2]
        x=Treinamento[1,:]
        y=Treinamento[2,:]
        z=Teste[1,:]
        w=Teste[2,:]
        for j=1:max_p

            #acha o erro para o conjunto de treinamento e depois para o conjunto de teste, para cada grau de polinômio

            beta=reg_poly(x,y,j)
            errinhoTR=zeros(1, length(x))

            for k=1:length(x)
                errinhoTR[k]=y[k]-beta[1]
                for h=2:j+1
                    errinhoTR[k]=errinhoTR[k]-beta[h]*x[k]^(h-1)
                end
                errinhoTR[k]=errinhoTR[k]^2
            end
            ErroTR_[i,j]=(norm(errinhoTR)^2)/(2*length(x))

            #----

            errinhoTE=zeros(1, length(z))
            for k=1:length(z)
                errinhoTE[k]=w[k]-beta[1]
                for h=2:j+1
                    errinhoTE[k]=errinhoTE[k]-beta[h]*z[k]^(h-1)
                end
                errinhoTE[k]=errinhoTE[k]^2
            end
            ErroTE_[i,j]=(norm(errinhoTE)^2)/(2*length(z))
        end
    end

    #mesmo código, só uma continuação do for
    for i=r+1:num_folds
        Teste = separa_matriz(dados_emb,1+(i-1)*(tam_conj),1+i*(tam_conj))[1]
        Treinamento = separa_matriz(dados_emb,1+(i-1)*(tam_conj),1+i*(tam_conj))[2]
        x=Treinamento[1,:]
        y=Treinamento[2,:]
        z=Teste[1,:]
        w=Teste[2,:]
        for j=1:max_p
            beta=reg_poly(x,y,j)
            errinhoTR=zeros(1, length(x))
            errinhoTE=zeros(1, length(z))
            for k=1:length(x)
                errinhoTR[k]=y[k]-beta[1]
                for h=2:j+1
                    errinhoTR[k]=errinhoTR[k]-beta[h]*x[k]^(h-1)
                end
                errinhoTR[k]=errinhoTR[k]^2
            end
            Erro_TR[i,j]=(norm(errinhoTR)^2)/(2*length(x))
            for k=1:length(z)
                errinhoTE[k]=w[k]-beta[1]
                for h=2:j+1
                    errinhoTE[k]=errinhoTE[k]-beta[h]*z[k]^(h-1)
                end
                errinhoTE[k]=errinhoTE[k]^2
            end
            Erro_TE[i,j]=(norm(errinhoTE)^2)/(2*length(z))
        end
    end

    #gráficos para cada fold; fixe j entre 1 e num_folds:
    v = zeros(1,max_p)
    for i=1:max_p
        v[1,i]=i;
    end

    j=1;



    scatter(v[1,:], Erro_TR[j,:], c= :red)
    scatter!(v[1,:], Erro_TE[j,:], c= :blue, yaxis=[0,0.001])

    #gráfico das medias

    Media_TR=zeros(1,max_p)
    Media_TE=zeros(1,max_p)

    for i=1:max_p
        Media_TR[1,i] = mean(Erro_TR[:,i])
        Media_TE[1,i] = mean(Erro_TE[:,i])
    end

    scatter(v[1,:], Media_TR[1,:], c= :red)
    scatter!(v[1,:], Media_TE[1,:], c=:blue, yaxis=[0,0.001])

    png("kfold")

    return Erro_TR, Erro_TE

end

main()

